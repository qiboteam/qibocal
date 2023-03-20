import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

from . import utils


def hyperopt(x_train, y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    clf = SVC(gamma="auto")

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = {}
    space["C"] = np.linspace(0.01, 2, num=50)
    space["degree"] = [2, 3, 4]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)

    return search.best_params_


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return SVC(gamma="auto").set_params(**hyperpars)


normalize = utils.scikit_normalize
