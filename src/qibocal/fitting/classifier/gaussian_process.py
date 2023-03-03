from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from . import utils


def constructor(**hyperparams):
    return GaussianProcessClassifier(1.0 * RBF(1.0)).set_params(**hyperparams)


def hyperopt(_x_train, _y_train, _path):
    return constructor().get_params()


normalize = utils.scikit_normalize
