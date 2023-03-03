from sklearn.svm import SVC

from . import utils

def constructor(**_hyperpars):
    return SVC(kernel="linear", C=0.025)

def hyperopt(_x_train, _y_train, _path):
    return constructor().get_params()

normalize = utils.scikit_normalize
