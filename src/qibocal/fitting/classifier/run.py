import importlib
import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qibolab.platforms.abstract import Qubit
from sklearn.metrics import accuracy_score

from . import data

CLS_MODULES = [
    "linear_svm",
    "ada_boost",
    "gaussian_process",
    "naive_bayes",
    "nn",
    "qubit_fit",
    "random_forest",
    "rbf_svm",
    "qblox_fit",
]


HYPERFILE = "hyperpars.json"
PREDFILE = "predictions.npy"
BENCHTABFILE = "benchmarks.csv"

base_dir = pathlib.Path()


def import_classifiers(cls_names: List[str]):
    r"""Return the classification models.

    Args:
        cls_name (list[str]): List of models' names.
    """
    importing_func = lambda mod: importlib.import_module(
        ".." + mod, "qibocal.fitting.classifier.*"
    )
    classifiers = list(map(importing_func, cls_names))
    return classifiers


class Classifier:
    r"""Classs to define the different classifiers used in the benchmarking.

    Args:
        mod: Classsification model.
        base_dir (Path): Where to store the classification results.

    """

    def __init__(self, mod, base_dir: pathlib.Path) -> None:
        self.mod = mod
        self.base_dir = base_dir
        self.trainable_model = None

    @property
    def name(self):
        r"""Model's name."""
        return self.mod.__name__.split(".")[-1]

    @property
    def hyperopt(self):
        r"""The function that performs the hyperparameters optimization."""
        return self.mod.hyperopt

    @property
    def normalize(self):
        r"""The function that adds a data normalisation
        stage before the model classification.
        """
        return self.mod.normalize

    @property
    def constructor(self):
        r"""The model builder."""
        return self.mod.constructor

    @property
    def fit(self):
        r"""The model's fitting function."""
        return self.mod.fit

    @property
    def savedir(self):
        r"""The saving path."""
        return self.base_dir / self.name

    @property
    def hyperfile(self):
        r"""The path where the hyperparameters are stored."""
        return self.savedir / HYPERFILE

    def dump(self, path: pathlib.Path):
        self.mod.dump(self.trainable_model, path)

    @classmethod
    def load_model(cls, name: str, base_dir: pathlib.Path):
        r"""
        Giving the classification name this method returns the respective classifier.
        Args:
            name (str): classifier's name.
            base_dir (path): Where to store the classification results.
        Returns:
            Classification model.
        """
        inst = cls(import_classifiers([name]), base_dir)
        hyperpars = inst.load_hyper()
        return inst.create_model(hyperpars)

    @classmethod
    def model_from_dir(cls, folder: pathlib.Path):
        name = folder.name
        base_dir = folder.parent
        return cls.load_model(name, base_dir)

    def dump_hyper(self, hyperpars):
        r"""Saves the hyperparameters"""
        self.hyperfile.write_text(json.dumps(hyperpars, default=str), encoding="utf-8")

    def load_hyper(self):
        r"""Loads the hyperparameters and returns them."""
        return json.loads(self.hyperfile.load_text(encoding="utf-8"))

    def create_model(self, hyperpars):
        r"""Returns a model with the normalization stage.

        Args:
            hyperpars: Model's hyperparameters.

        Returns:
            Classification model.
        """
        self.trainable_model = self.normalize(self.constructor(hyperpars))
        return self.trainable_model


@dataclass
class BenchmarkResults:
    r"""Class that stores the models information."""
    accuracy: float
    testing_time: float
    training_time: float
    name: Optional[str] = None


def benchmarking(model, x_train, y_train, x_test, y_test, **fit_kwargs):
    r"""This function evaluates the model's performances.
    Args:
        model: Classification model with `fit` and `predict` methods.
        x_train: Training input.
        y_train: Training output.
        x_test: Test input.
        y_test: Test output.
        **fit_kwargs:  Arbitrary keyword arguments for the `fit` function.

    Returns:
        - results (BenchmarkResults): Stores the model's accuracy, the training and testing time.
        - y_pred: Model's predictions.
        - model: trained model.
        - fit_info: Stores training infos.
    """
    # Evaluate training time
    start = time.time()
    y_train = y_train.astype("int")
    fit_info = model.fit(x_train, y_train, **fit_kwargs)
    stop = time.time()
    training_time = stop - start
    # Evaluate test time per element
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    test_time = (stop - start) / len(x_test)
    # Evaluate accuracy
    y_pred = np.round(y_pred).tolist()

    score = accuracy_score(y_test.tolist(), y_pred)
    logging.info(f"Accuracy: {score}")
    results = BenchmarkResults(score, test_time, training_time)

    return results, y_pred, model, fit_info


def plot_history(history, save_dir: pathlib.Path):
    r"""Plots the neural network history
    and save it in `json` file.

    Args:
        history (keras.callbacks.History): History.
        save_dir (Path): Storing path.
    """
    history_dict = history.history
    epochs = len(history_dict["loss"])
    plt.figure(figsize=(14, 7))
    plt.plot(range(epochs), history_dict["loss"], label="loss")
    plt.plot(range(epochs), history_dict["accuracy"], label="accurancy")
    plt.plot(range(epochs), history_dict["val_loss"], label="val_loss")
    plt.plot(range(epochs), history_dict["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir / "NN_training.pdf")
    json.dump(history_dict, open(save_dir / "NN_history.json", "w"))


def train_qubit(
    qubit: Qubit,
    base_dir,
    qubits_data=None,
    classifiers=None,
):
    r"""Given a dataset `qubits_data` with qubits' information, this function performs the benchmarking of some classifiers.
    Each model's prediction `y_pred` is saved in  `basedir/qubit{qubit}/{classifier name}/predictions.npy`.

    Args:
        base_dir (path): Where save the results.
        qubit (int): Qubit ID.
        qubits_data (DataFrame): data about the qubits` states.
        classifiers (list | None, optional): List of classification models. It must be a subset of `CLS_MODULES`.

    Returns:
        - benchmarks_table (pd.DataFrame): Table with the following columns

            - **name**: model's name
            - **accuracy**: model's accuracy
            - **training_time**: training time in seconds
            - **testing_time**: testing time per item in seconds.

        - y_test (list): List of test outputs.
        - x_test (list): Tests inputs.
        - models (list): List of trained models.
        - Names (list): Models' names
        - hpars_list(list): Models' hyper-parameters.

    """

    qubit_data = qubits_data[qubits_data["qubit"] == qubit.name]
    nn_epochs = 200
    nn_val_split = 0.2
    qubit_dir = pathlib.Path(base_dir) / f"qubit{qubit.name}"
    qubit_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = data.generate_models(qubit_data)
    models = []
    results_list = []
    names = []
    hpars_list = []
    if classifiers is None:
        classifiers = CLS_MODULES

    classifiers = import_classifiers(classifiers)

    for mod in classifiers:
        classifier = Classifier(mod, qubit_dir)
        logging.info(f"Classification model: {classifier.name}")
        if classifier.name not in qubit.classifiers_hpars:
            hyperpars = classifier.hyperopt(
                x_train, y_train.astype(np.int64), classifier.savedir
            )
        else:
            hyperpars = qubit.classifiers_hpars[classifier.name]
        hpars_list.append(hyperpars)
        model = classifier.create_model(hyperpars)

        if classifier.name == "nn":
            results, y_pred, model, fit_info = benchmarking(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                epochs=nn_epochs,
                validation_split=nn_val_split,
            )
            plot_history(fit_info, classifier.savedir)
        else:
            results, y_pred, model, _ = benchmarking(
                model, x_train, y_train, x_test, y_test
            )
        models.append(model)  # save trained model
        results.name = classifier.name
        results_list.append(results)
        names.append(classifier.name)
        classifier.savedir.mkdir(exist_ok=True)
        classifier.dump_hyper(hyperpars)
        dump_preds(y_pred, classifier.savedir)
        classifier.dump(classifier.savedir / classifier.name)

    benchmarks_table = pd.DataFrame([asdict(res) for res in results_list])
    return benchmarks_table, y_test, x_test, models, names, hpars_list


def dump_preds(y_pred, dir_path):
    r"""Dumps the predictions in `{dir_path}/predictions.npy`.

    Args:
        y_pred (list): Predictions.
        dir_path (path): Saving path.
    """
    np.save(dir_path / PREDFILE, y_pred)


def dump_benchmarks_table(table, dir_path):
    r"""Dumps the benchmark table in `{dir_path}/benchmarks.csv`.

    Args:
        table (pd.DataFrame): Predictions.
        dir_path (path): Saving path.
    """
    table.to_csv(dir_path / BENCHTABFILE)


def preds_from_file(dir_path):
    r"""Load the predictions from a file

    Args:
        dir_path (path): Where the file `predictions.npy` is.

    Returns:
        Predictions.
    """
    return np.load(dir_path / PREDFILE)


def table_from_file(dir_path):
    r"""Load and return the benchmark table from a file

    Args:
        dir_path (path): Where the file `benchmarks.csv` is.
    """
    return pd.read_csv(dir_path / BENCHTABFILE)
