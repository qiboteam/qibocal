from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from . import utils


def constructor(_hyperparams):
    return GaussianProcessClassifier(1.0 * RBF(1.0))

<<<<<<< HEAD
def hyperopt(_x_train, _y_train, _path):
=======

def hyperopt(_x_train, _y_train):
>>>>>>> 7db286979b09c5392cbe222101f3eda72a7972bd
    return {}


normalize = utils.scikit_normalize
