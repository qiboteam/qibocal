from sklearn.naive_bayes import GaussianNB

from . import utils

<<<<<<< HEAD
def constructor(**hyperpars):
    return GaussianNB().set_params(**hyperpars)
=======

def constructor(**_hyperpars):
    return GaussianNB()
>>>>>>> 8c18893fa1c83d022618b12c2d8de4f212eb400b


def hyperopt(_x_train, _y_train, _path):
    return constructor().get_params()


normalize = utils.scikit_normalize
