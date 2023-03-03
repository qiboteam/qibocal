import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import run


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


def plot_conf_matr(y_test, model_list, base_dir: pathlib.Path):
    matrices = []
    names = []
    _figure = plt.figure(figsize=(30, 5))

    for count, model in enumerate(model_list):
        classifier = run.Classifier(model, base_dir)
        names.append(classifier.name)
        y_pred = np.load(base_dir / classifier.name / run.PREDFILE)
        conf_matr = confusion_matrix(y_test, y_pred, normalize="true")
        matrices.append(conf_matr)
        ax = plt.subplot(1, len(model_list), count + 1)
        sns.heatmap(
            conf_matr, annot=True, xticklabels=["P", "N"], yticklabels=["P", "N"]
        )
        ax.set_title(classifier.name)

    plt.savefig(base_dir / "confusion_matrices.pdf")
    confusion_dic = {names[i]: matrices[i].tolist() for i in range(len(model_list))}
    json.dump(confusion_dic, open(base_dir / "confusion_matrices.json", "w"))
