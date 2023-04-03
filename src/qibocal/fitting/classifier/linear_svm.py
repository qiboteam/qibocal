from sklearn.svm import SVC

from . import utils

REG_PARAM = 0.025


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        hyperparams: Model hyperparameters.
    """
    return SVC(kernel="linear", C=REG_PARAM).set_params(**hyperpars)


def hyperopt(_x_train, _y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        _x_train: Training inputs.
        _y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    model = SVC(kernel="linear", C=REG_PARAM)
    return model.get_params()


normalize = utils.scikit_normalize
