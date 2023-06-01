import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from . import scikit_utils


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return RandomForestClassifier().set_params(**hyperpars)


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
    space = {}
    space["n_estimators"] = np.arange(10, 200, 10, dtype=int)
    space["criterion"] = ["gini", "entropy", "log_loss"]
    space["max_features"] = ["sqrt", "log2", None]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train.tolist())
    return search.best_params_


normalize = scikit_utils.scikit_normalize
dump = scikit_utils.scikit_dump
predict_from_file = scikit_utils.scikit_predict
