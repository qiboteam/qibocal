from sklearn.naive_bayes import GaussianNB

from . import utils


def constructor(hyperpars):
    return GaussianNB().set_params(**hyperpars)


def hyperopt(_x_train, _y_train, _path):
    model = GaussianNB()
    return model.get_params()


normalize = utils.scikit_normalize
