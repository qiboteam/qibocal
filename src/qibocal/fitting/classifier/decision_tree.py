from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import scikit_utils


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        hyperparams: Model hyperparameters.
    """
    return DecisionTreeClassifier().set_params(**hyperpars)


def hyperopt(x_train, y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    clf = DecisionTreeClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = {}
    space["criterion"] = ["gini", "entropy", "log_loss"]
    space["splitter"] = ["best", "random"]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)

    return search.best_params_


normalize = scikit_utils.scikit_normalize
dump = scikit_utils.scikit_dump
predict_from_file = scikit_utils.scikit_predict
