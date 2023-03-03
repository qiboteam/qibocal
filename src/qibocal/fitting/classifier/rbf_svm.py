import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC


def hyperopt(x_train, y_train, _path):
    clf = SVC(gamma="auto")

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space["C"] = np.linspace(0.01, 2, num=50)
    space["degree"] = [2, 3, 4]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)

    return search.best_params_


def constructor(**hyperpars):
    return SVC(gamma="auto", **hyperpars)


normalize = utils.scikit_normalize
