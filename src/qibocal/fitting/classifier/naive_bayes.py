from sklearn.naive_bayes import GaussianNB

from . import utils


def constructor(**_hyperpars):
    return GaussianNB()


def hyperopt(_x_train, _y_train, _path):
    return constructor().get_params()


normalize = utils.scikit_normalize
