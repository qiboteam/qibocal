import enum
import json
import pathlib
import time
from dataclasses import dataclass, asdict
from sklearn.metrics import confusion_matrix
from typing import Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

from . import ( nn, rbf_svm, naive_bayes, 
               ada_boost, gaussian_process, 
               linear_svm, random_forest )
from . import plots
from . import data

class Classifiers(enum.Enum):
    #nn = nn
    linear_svm = linear_svm
    naive_bayes = naive_bayes
    rbf_svm = rbf_svm

HYPERFILE = "hyperpars.json"
PREDFILE = "predictions.npy"
BENCHTABFILE = "benchmarks.csv"

base_dir = pathlib.Path()

class Classifier():
    def __init__(self, mod, base_dir: pathlib.Path) -> None:
        self.mod = mod
        self.base_dir = base_dir

    @property
    def name(self):
        return self.mod.__name__.split(".")[-1]
            
    @property
    def hyperopt(self):
        return self.mod.hyperopt

    @property
    def normalize(self):
        return self.mod.normalize
    
    @property
    def constructor(self):
        return self.mod.constructor
    
    @property
    def fit(self):
        return self.mod.fit
    
    @property
    def plots(self):
        self.mod.plots()


    @property
    def savedir(self):
        return self.base_dir / self.name


    @property
    def hyperfile(self):
        return self.savedir / HYPERFILE

    @classmethod
    def load_model(cls, name: str, base_dir: pathlib.Path):
        inst = cls(Classifiers[name], base_dir)
        hyperpars = inst.load_hyper()
        return inst.create_model(hyperpars)
    
    @classmethod
    def model_from_dir(cls, folder: pathlib.Path):
        name = folder.name
        base_dir = folder.parent
        return cls.load_model(name, base_dir)

    def dump_hyper(self, hyperpars):
        self.hyperfile.write_text(json.dumps(hyperpars), encoding="utf-8")
    
    def load_hyper(self):
        return json.loads(self.hyperfile.load_text(encoding="utf-8"))

    def create_model(self, hyperpars):
        return self.normalize(self.constructor(**hyperpars))
    
@dataclass
class BenchmarkResults:
    accuracy: float
    testing_time: float
    training_time: float
    name: Optional[str] = None
    
def benchmarking(model, x_train, y_train, x_test, y_test, fit_kwargs=None):
    if fit_kwargs is None:
        fit_kwargs = {}
    
    start = time.time()
    fit_info = model.fit(x_train, y_train, **fit_kwargs)
    stop = time.time()
    training_time = stop-start
    score = model.score(x_test, y_test)
    print("Accuracy", score)
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    test_time = (stop - start)/len(x_test)

    results = BenchmarkResults(score, test_time, training_time)
    
    return results, y_pred, fit_info


def plot_history(epochs,history, save_dir):
    history_dict = history.history
    plt.figure(figsize=(14,7))
    plt.plot( range(epochs),history_dict["loss"],label="loss")
    plt.plot( range(epochs),history_dict["accuracy"],label='accurancy')
    plt.plot( range(epochs),history_dict["val_loss"],label='val_loss')
    plt.plot( range(epochs),history_dict["val_accuracy"],label='val_accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(save_dir / "NN_training.pdf")
    json.dump( history_dict, open( save_dir / "NN_history.json", 'w' ) )



def train_qubit(data_path, base_dir: pathlib.Path, qubit):
    nn_epochs = 200
    nn_val_split = 0.2
    qubit_dir = base_dir / f"qubit{qubit}"
    qubit_dir.mkdir()
    qubit_data = data.load_qubit(data_path, qubit)
    data.plot_qubit(qubit_data, qubit_dir)
    x_train,y_train,x_test,y_test = data.generate_models(qubit_data)
    models = []
    results_list = []
    conf_matrices = []
    
    for modvariant in Classifiers:
        mod = modvariant.value
        classifier = Classifier(mod, qubit_dir)
        classifier.savedir.mkdir()
        hyperpars = classifier.hyperopt(x_train,y_train, classifier.savedir)
        classifier.dump_hyper(hyperpars)
        print(hyperpars)
        model = classifier.create_model(hyperpars)
        models.append(model)

        if model is nn :
            results, y_pred, fit_info = benchmarking(model, x_train, y_train, x_test, y_test, epochs=nn_epochs, validation_split=nn_val_split)
            plot_history(nn_epochs, fit_info, classifier.savedir)
        else:
            results, y_pred, _ = benchmarking(model, x_train, y_train, x_test, y_test)
        
        results.name = classifier.name
        results_list.append(results)
        conf_matrices.append(confusion_matrix(y_test,y_pred, normalize='true'))
        
        dump_preds(y_pred, classifier.savedir)

    benchmarks_table = pd.DataFrame([asdict(res) for res in results_list])
    return benchmarks_table

def dump_preds(y_pred, dir_path):
    np.save(dir_path / PREDFILE, y_pred)

def dump_benchmarks_table(table, dir_path):
    table.to_csv(dir_path / BENCHTABFILE)

def preds_from_file(dir_path):
    return np.load(dir_path / PREDFILE)

def table_from_file(dir_path):
    return pd.read_csv(dir_path / BENCHTABFILE)

# def plot_qubit(folder: pathlib.Path):
#     data = load_data()
#     plots.data(data)

#     models = []
#     for model_dir in folder.glob("*"):
#         model = Classifier.load_model(model_dir)
#     plots.common(models)
