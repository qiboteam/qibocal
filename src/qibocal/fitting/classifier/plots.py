import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve

from . import run

TABLEFILE = "benchmarks.pdf"
CONFMATRIXFIG = "confusion_matrices.pdf"
CONFMATRIXFILE = "confusion_matrices.json"
ROCFIG = "ROC_curves.pdf"
ROCFILE = "ROC_curves.json"
RESULTSFIG = "results.pdf"
COLOR = ListedColormap(["#FF0000", "#0000FF"])


def plot_table(table, path: pathlib.Path):
    r"""Plot the benchmark table and save it as
    `{path}/benchmarks.pdf`.

    Args:
        table (pd.DataFrame): Benchmark's table.
        path (path): Save path.
    """
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
    plt.tight_layout()
    plt.savefig(path / TABLEFILE)


def plot_conf_matr(y_test, path: pathlib.Path, classifiers=None):
    r"""
    Plot the confusion matrices of the `classifiers` and save it
    as  `{path}/confusion_matrices.pdf` and `{path}/confusion_matrices.json`.

    Args:
        y_test: Test outputs.
        path: Save path.
        classifiers (list | None, optional): List of classification models. It must be a subset of `run.CLS_MODULES`.
    """
    matrices = []
    names = []
    _figure = plt.figure(figsize=(30, 5))

    if classifiers is None:
        classifiers = run.CLS_MODULES

    classifiers = run.import_classifiers(classifiers)

    for count, model in enumerate(classifiers):
        classifier = run.Classifier(model, path)
        names.append(classifier.name)

        y_pred = np.load(path / classifier.name / run.PREDFILE)
        # Evaluate confusion matrices
        conf_matr = confusion_matrix(y_test, np.round(y_pred), normalize="true")
        matrices.append(conf_matr)
        # Plots
        ax = plt.subplot(1, len(classifiers), count + 1)
        sns.heatmap(
            conf_matr, annot=True, xticklabels=["P", "N"], yticklabels=["P", "N"]
        )
        ax.set_title(classifier.name)

    plt.tight_layout()
    plt.savefig(path / CONFMATRIXFIG)
    confusion_dic = {names[i]: matrices[i].tolist() for i in range(len(classifiers))}
    # Save data in JSON
    json.dump(
        confusion_dic,
        open(path / CONFMATRIXFILE, "w"),
        separators=(",", ":"),
        sort_keys=True,
        indent=4,
    )


def plot_roc_curves(x_test, y_test, path: pathlib.Path, models, models_names):
    r"""Plot the ROC curves of the `models` and save it
    as `{path}/ROC_curves.pdf` and `{path}/ROC_curves.json`.
    Args:
        x_test (list): Test inputs.
        y_test (list): Test outputs.
        path (path): Save path.
        models (list): List of trained classifiers.
        models_names (list[str]): List of classifiers' names.
    """
    _figure = plt.figure(figsize=(30, 5))
    fprs = []
    tprs = []

    len_list = len(models)
    for count, model in enumerate(models):
        ax = plt.subplot(1, len_list, count + 1)
        plt.subplot(1, len_list, count + 1)

        # Evaluate the ROC curves
        y_pred = np.load(path / models_names[count] / run.PREDFILE).tolist()
        fpr, tpr, _ = roc_curve(y_test.tolist(), y_pred)
        fprs.append(fpr)
        tprs.append(tpr)

        try:
            RocCurveDisplay.from_estimator(
                model, x_test, y_test, ax=ax, color="darkorange"
            )
        except ValueError:
            RocCurveDisplay.from_predictions(
                y_test.tolist(),
                y_pred,
                ax=ax,
                color="darkorange",
            )
        # Plot
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{models_names[count]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path / ROCFIG)

    # Save data in a dictionary
    roc_dict = {
        models_names[i]: [tprs[i].tolist(), fprs[i].tolist()] for i in range(len_list)
    }
    json.dump(
        roc_dict,
        open(path / ROCFILE, "w"),
        separators=(",", ":"),
        sort_keys=True,
        indent=4,
    )


def plot_models_results(x_train, x_test, y_test, path, models, models_names):
    r"""Plot the decisions boundaries of the `c` and save it
    as `{path}/results.pdf`.
    Args:
        x_train (list): Train inputs.
        x_test (list): Test inputs.
        y_test (list): Test outputs.
        path (path): Save path.
        models (list): List of trained classifiers.
        models_names (list[str]): List of classifiers' names.
    """
    _figure = plt.figure(figsize=(20, 8))

    len_list = len(models)

    for count, classifier in enumerate(models):
        ax = plt.subplot(3, len_list // 3 + 1, count + 1)

        i, q = np.meshgrid(
            np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), num=200),
            np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), num=200),
        )
        grid = np.vstack([i.ravel(), q.ravel()]).T
        y_pred = np.reshape(classifier.predict(grid), q.shape)
        display = DecisionBoundaryDisplay(xx0=i, xx1=q, response=y_pred)

        display.plot(cmap="RdBu", alpha=0.8, ax=ax)
        cm_bright = COLOR
        ax.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
        )

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(models_names[count])
        plt.tight_layout()
        plt.savefig(path / RESULTSFIG)
