import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

from . import scikit_utils


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        hyperparams: Model hyperparameters.
    """
    return MLPClassifier().set_params(**hyperpars)


def hyperopt(x_train, y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    clf = MLPClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = {}
    space["hidden_layer_sizes"] = [(16, 16), (16, 32), (16, 64)]
    space["learning_rate_init"] = np.linspace(1e-4, 1e-2, num=3)
    space["activation"] = ["logistic", "tanh", "relu"]
    space["solver"] = ["lbfgs", "sgd", "adam"]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)

    return search.best_params_


normalize = lambda x: x
dump = scikit_utils.scikit_dump
predict_from_file = scikit_utils.scikit_predict
