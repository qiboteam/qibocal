from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from . import utils


def constructor(hyperparams):
    return GaussianProcessClassifier(1.0 * RBF(1.0)).set_params(**hyperparams)


def hyperopt(_x_train, _y_train, _path):
    # Build the best model
    model = GaussianProcessClassifier(1.0 * RBF(1.0))
    return model.get_params()


normalize = utils.scikit_normalize
