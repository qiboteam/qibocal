from sklearn.svm import SVC

from . import utils


def constructor(**hyperpars):
    return SVC(kernel="linear", C=0.025).set_params(**hyperpars)


def hyperopt(_x_train, _y_train, _path):
    return constructor().get_params()


normalize = utils.scikit_normalize
