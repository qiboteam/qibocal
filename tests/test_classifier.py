import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from qibocal.fitting.classifier import plots, run


@pytest.fixture
def data_creation():
    state0_center = [0.0, 0.0]
    state1_center = [2.0, 2.0]
    cov = 0.1 * np.eye(2)
    size = 100

    data_0 = np.random.multivariate_normal(state0_center, cov, size=size)
    data_1 = np.random.multivariate_normal(state1_center, cov, size=size)
    data = {
        "i": data_0[:, 0].tolist() + data_1[:, 0].tolist(),
        "q": data_0[:, 1].tolist() + data_1[:, 1].tolist(),
        "state": [0] * size + [1] * size,
        "qubit": [0, 1] * size,
    }
    data = pd.DataFrame(data)
    data_path = tempfile.mkstemp(suffix=".csv")[1]
    data.to_csv(data_path)
    return data_path


@pytest.fixture
def initialization(data_creation):
    data_path = data_creation
    data_path = pathlib.Path(data_path)
    base_dir = pathlib.Path(tempfile.mkdtemp())
    qubits = [0, 1]
    for qubit in qubits:
        qubit_dir = base_dir / f"qubit{qubit}"
        print(base_dir)
        classifiers = ["linear_svm"]
        table, y_test, _x_test = run.train_qubit(
            data_path, base_dir, qubit=qubit, classifiers=classifiers
        )
        run.dump_benchmarks_table(table, qubit_dir)
        plots.plot_table(table, qubit_dir)
        plots.plot_conf_matr(y_test, qubit_dir, classifiers=classifiers)
    return table, base_dir, y_test


def test_folders(initialization):
    qubits = [0, 1]
    _, base_dir, _ = initialization
    files = [
        "benchmarks.pdf",
        "benchmarks.csv",
        "confusion_matrices.json",
        "confusion_matrices.pdf",
    ]
    for qubit in qubits:
        qubit_dir = base_dir / f"qubit{qubit}"
        assert os.path.exists(qubit_dir)
        for file in files:
            assert os.path.exists(qubit_dir / file)


def test_accuracy(initialization):
    table, _, y_test = initialization
    real_accuracies = table["accuracy"].tolist()
    # The model is evaluated good if it
    # misclassifys less than two points
    min_accuracy = 1 - 2.0 / len(y_test)

    for acc in real_accuracies:
        assert acc > min_accuracy
