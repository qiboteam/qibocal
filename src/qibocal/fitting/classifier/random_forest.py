import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from . import utils

def hyperopt(x_train,y_train, _path):
    clf = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space["n_estimators"] = np.linspace(10, 200, num=20).astype("int")
    space["criterion"] = ["gini", "entropy", "log_loss"]
    space["max_features"] = ["sqrt", "log2", None]
    search = GridSearchCV(clf, space, scoring="accuracy", n_jobs=-1, cv=cv)
    _ = search.fit(x_train, y_train)
    return search.best_params_


def constructor(hyperpars):
    return RandomForestClassifier().set_params(**hyperpars)

normalize = utils.scikit_normalize
