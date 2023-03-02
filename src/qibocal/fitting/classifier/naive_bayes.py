from sklearn.naive_bayes import GaussianNB

from . import utils

constructor = GaussianNB()


def hyperopt(_x_train, _y_train) -> HyperPars:
    return constructor.get_params()


normalize = utils.scikit_normalize(constructor)

# def normalize(unn):
#     ...
#     return alkjclksjl
