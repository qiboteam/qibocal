from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from . import utils

def constructor(_hyperparams):
    return GaussianProcessClassifier(1.0 * RBF(1.0))

def hyperopt(_x_train, _y_train):
    return {}

normalize = utils.scikit_normalize
