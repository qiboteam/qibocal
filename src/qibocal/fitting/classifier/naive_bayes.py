from sklearn.naive_bayes import GaussianNB

from . import utils


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        hyperparams: Model hyperparameters.
    """
    return GaussianNB().set_params(**hyperpars)


def hyperopt(_x_train, _y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        _x_train: Training inputs.
        _y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    model = GaussianNB()
    return model.get_params()


normalize = utils.scikit_normalize
