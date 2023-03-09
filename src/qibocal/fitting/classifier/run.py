import enum
import importlib
import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from . import data, plots

CLS_MODULES = [
    "linear_svm",
    "ada_boost",
    "gaussian_process",
    "naive_bayes",
    "nn",
    "qubit_fit",
    "random_forest",
    "rbf_svm",
]


HYPERFILE = "hyperpars.json"
PREDFILE = "predictions.npy"
BENCHTABFILE = "benchmarks.csv"

base_dir = pathlib.Path()


def import_classifiers(cls_names: list[str]):
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
        inst = cls(Classifiers[name], base_dir)
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
        return self.normalize(self.constructor(hyperpars))


@dataclass
class BenchmarkResults:
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
    fit_info = model.fit(x_train, y_train, **fit_kwargs)
    stop = time.time()
    training_time = stop - start
    # Evaluate test time per element
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    test_time = (stop - start) / len(x_test)
    # Evaluate accuracy
    score = accuracy_score(y_test, np.round(y_pred))
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
    data_path: pathlib.Path, base_dir: pathlib.Path, qubit, classifiers=None
):
    r"""Given a dataset in `data_path` with qubits' information, this function performs the benchmarking of some classifiers.
    Each model's prediction `y_pred` is saved in  `basedir/qubit{qubit}/{classifier name}/predictions.npy`.

    Args:
        data_path(path): Where the qubits' data are stored.
        base_dir (path): Where save the results.
        qubit (int): Qubit ID.
        classifiers (list | None, optional): List of classification models. It must be a subset of `CLS_MODULES`.

    Returns: benchmarks_table, y_test
        - benchmarks_table (pd.DataFrame): Table with the following columns

            - **name**: model's name
            - **accuracy**: model's accuracy
            - **training_time**: training time in seconds
            - **testing_time**: testing time per item in seconds.

        - y_test (list): List of test outputs.
    """
    nn_epochs = 200
    nn_val_split = 0.2
    qubit_dir = base_dir / f"qubit{qubit}"
    qubit_dir.mkdir(exist_ok=True)
    qubit_data = data.load_qubit(data_path, qubit)
    data.plot_qubit(qubit_data, qubit_dir)
    x_train, y_train, x_test, y_test = data.generate_models(qubit_data)
    models = []
    results_list = []
    names = []
    if classifiers is None:
        classifiers = CLS_MODULES  # [i.value for i in Classifiers]

    classifiers = import_classifiers(classifiers)

    for mod in classifiers:
        classifier = Classifier(mod, qubit_dir)
        classifier.savedir.mkdir(exist_ok=True)
        logging.info(f"Classification model: {classifier.name}")
        hyperpars = classifier.hyperopt(x_train, y_train, classifier.savedir)

        classifier.dump_hyper(hyperpars)
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
        dump_preds(y_pred, classifier.savedir)

    benchmarks_table = pd.DataFrame([asdict(res) for res in results_list])
    plots.plot_models_results(x_train, x_test, y_test, qubit_dir, models, names)
    plots.plot_roc_curves(x_test, y_test, qubit_dir, models, names)

    return benchmarks_table, y_test, x_test


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
    r"""Load the benchmark table from a file

    Args:
        dir_path (path): Where the file `benchmarks.csv` is.

    Returns:

    """
    return pd.read_csv(dir_path / BENCHTABFILE)
