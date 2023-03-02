from sklearn.svm import SVC

from . import utils

constructor = SVC(kernel="linear", C=0.025)


def hyperopt(_x_train, _y_train):
    return constructor.get_params()


normalize = utils.scikit_normalize(constructor)
