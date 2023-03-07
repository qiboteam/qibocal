import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve

from . import run
from .run import Classifiers


def plot_table(table, path):
    sns.set_style("darkgrid")
    g = sns.PairGrid(
        table,
        y_vars="name",
        x_vars=["accuracy", "testing_time", "training_time"],
        height=4,
        hue="name",
        palette="bright",
    )
    g.map(sns.scatterplot)
    plt.xscale("log")
    plt.savefig(path / "benchmarks.pdf")


def plot_conf_matr(y_test, base_dir: pathlib.Path, classifiers=None):
    matrices = []
    names = []
    _figure = plt.figure(figsize=(30, 5))

    if classifiers is None:
        classifiers = [i.value for i in Classifiers]

    for count, model in enumerate(classifiers):
        classifier = run.Classifier(model, base_dir)
        names.append(classifier.name)
        y_pred = np.load(base_dir / classifier.name / run.PREDFILE)
        conf_matr = confusion_matrix(y_test, np.round(y_pred), normalize="true")
        matrices.append(conf_matr)
        ax = plt.subplot(1, len(classifiers), count + 1)
        sns.heatmap(
            conf_matr, annot=True, xticklabels=["P", "N"], yticklabels=["P", "N"]
        )
        ax.set_title(classifier.name)

    plt.savefig(base_dir / "confusion_matrices.pdf")
    confusion_dic = {names[i]: matrices[i].tolist() for i in range(len(classifiers))}

    json.dump(
        confusion_dic,
        open(base_dir / "confusion_matrices.json", "w"),
        separators=(",", ":"),
        sort_keys=True,
        indent=4,
    )


def plot_roc_curves(x_test, y_test, base_dir: pathlib.Path, models, names):
    _figure = plt.figure(figsize=(30, 5))
    fprs = []
    tprs = []
    # names = []

    # if classifiers is None:
    #     classifiers = [i.value for i in Classifiers]

    # len_list = len(classifiers)
    len_list = len(models)
    for count, model in enumerate(models):
        # classifier = run.Classifier(model, base_dir)
        # names.append(names[count])

        ax = plt.subplot(1, len_list, count + 1)
        plt.subplot(1, len_list, count + 1)

        y_pred = np.load(base_dir / names[count] / run.PREDFILE)
        fpr, tpr, _ = roc_curve(
            y_test, y_pred
        )  # TODO: add dictionary and Ramiro's method and close windows with plt.close('all')
        fprs.append(fpr)
        tprs.append(tpr)

        print(model)
        try:
            RocCurveDisplay.from_estimator(
                model, x_test, y_test, ax=ax, color="darkorange"
            )
        except:
            # roc_auc = auc(fpr, tpr)
            # _roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, ax=ax,color="darkorange")
            RocCurveDisplay.from_predictions(
                y_test,
                y_pred,
                ax=ax,
                color="darkorange",
            )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{names[count]}")
        plt.legend()
        plt.savefig(base_dir / "ROC_curves.pdf")

    roc_dict = {names[i]: [tprs[i].tolist(), fprs[i].tolist()] for i in range(len_list)}

    json.dump(
        roc_dict,
        open(base_dir / "roc_curves.json", "w"),
        separators=(",", ":"),
        sort_keys=True,
        indent=4,
    )
    plt.close("all")


def plot_models_results(
    x_train, x_test, y_test, base_dir, classifiers, classifiers_name
):
    i = 1
    _figure = plt.figure(figsize=(20, 8))

    len_list = len(classifiers)

    for count, classifier in enumerate(classifiers):
        # classifier = run.Classifier(model, base_dir)

        ax = plt.subplot(3, len_list // 3 + 1, count + 1)

        i, q = np.meshgrid(
            np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), num=200),
            np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), num=200),
        )
        grid = np.vstack([i.ravel(), q.ravel()]).T
        y_pred = np.reshape(classifier.predict(grid), q.shape)
        display = DecisionBoundaryDisplay(xx0=i, xx1=q, response=y_pred)

        cm = plt.cm.RdBu
        display.plot(cmap=cm, alpha=0.8, ax=ax)
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
        )

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(classifiers_name[count])
        plt.savefig(base_dir / "results.pdf")
