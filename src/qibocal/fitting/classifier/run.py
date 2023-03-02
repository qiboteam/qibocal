import json
import pathlib
import time 
from dataclasses import dataclass
from typing import Any

from . import nn, rbf_svm, naive_bayes
from . import plots
from . import data

CLASSIFIERS = [nn, rbf_svm, naive_bayes]
HYPERFILE = "hyperpars.json"

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
    def normalize(self, model):
        return self.mod.normalize(model)
    
    @property
    def constructor(self):
        return self.mod.constructor
    
    @property
    def fit(self):
        return self.mod.fit
    
    @property
    def plots(self):
        self.mod.plots()

    def hyperfile(self):
        return self.base_dir / self.name / HYPERFILE

    @classmethod
    def load_model(cls, name: str, base_dir: pathlib.Path):
        inst = cls(CLASSIFIERS[name], base_dir)
        hyperpars = inst.load_hyper()
        return inst.create_model(hyperpars)
    
    @classmethod
    def model_from_dir(cls, folder: pathlib.Path):
        name = folder.basename()
        base_dir = folder.dirname()
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
    # return confusion_matrix(y_test,y_pred,normalize="true"), fit_info

pd.DataFrame([asdict(res) for res in results_list])

def train_qubit(data_path, save_dir, qubit, base_dir,  filter_):
    
    qubit_dir = save_dir / f"qubit{qubit}"
    qubit_dir.mkdir()
    qubit_data = data.load_qubit(data_path, qubit)
    data.plot_qubit(qubit_dir, qubit_data)
    x_train,y_train,x_test,y_test = data.generate_models(qubit_data, save_dir = qubit_dir)
    models = []

    for mod in CLASSIFIERS:
        classifier = Classifier(mod, base_dir)

        hyperpars = classifier.hyperopt(x_train,y_train)
        classifier.dump_hyper(hyperpars)

        model = classifier.create_model(hyperpars)
        models.append(model)


def plot_qubit(folder: pathlib.Path):
    data = load_data()
    plots.data(data)

    models = []
    for model_dir in folder.glob("*"):
        model = Classifier.load_model(model_dir)
    plots.common(models)
