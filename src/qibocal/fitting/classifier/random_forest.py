import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

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
    clf = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space["n_estimators"] = np.arange(10, 200, 10, dtype=int)
    space["criterion"] = ["gini", "entropy", "log_loss"]
    space["max_features"] = ["sqrt", "log2", None]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)
    return search.best_params_


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return RandomForestClassifier().set_params(**hyperpars)


normalize = utils.scikit_normalize
