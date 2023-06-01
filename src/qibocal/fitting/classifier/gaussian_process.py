from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from . import scikit_utils


def constructor(hyperparams):
    r"""Return the model class.

    Args:
        hyperparams: Model hyperparameters.
    """
    return GaussianProcessClassifier(1.0 * RBF(1.0)).set_params(**hyperparams)


def hyperopt(_x_train, _y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        _x_train: Training inputs.
        _y_train: Training outputs.
        _path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    # Build the best model
    model = GaussianProcessClassifier(1.0 * RBF(1.0))
    return model.get_params()


normalize = scikit_utils.scikit_normalize
dump = scikit_utils.scikit_dump
predict_from_file = scikit_utils.scikit_predict
