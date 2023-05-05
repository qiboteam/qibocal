import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from qibocal.fitting.classifier import plots, run

FILES_NAME = [
    "benchmarks.pdf",
    "benchmarks.csv",
    "confusion_matrices.json",
    "confusion_matrices.pdf",
]


@pytest.fixture
def data_path(tmp_path):
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
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path)
    return data_path


def qubit_path(base_dir: pathlib.Path, qubit):
    return base_dir / f"qubit{qubit}"


def initialization(data_path):
    base_dir = pathlib.Path(tempfile.mkdtemp())
    qubits = [0, 1]
    for qubit in qubits:
        qubit_dir = qubit_path(base_dir, qubit)
        classifiers = ["linear_svm", "qblox_fit"]
        table, y_test, _x_test = run.train_qubit(
            data_path, base_dir, qubit=qubit, classifiers=classifiers
        )
        run.dump_benchmarks_table(table, qubit_dir)
        plots.plot_table(table, qubit_dir)
        plots.plot_conf_matr(y_test, qubit_dir, classifiers=classifiers)
    return table, base_dir, y_test


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="no tensorflow-io-0.32.0's wheel available for Windows",
)
def test_folders(data_path):
    qubits = [0, 1]
    _, base_dir, _ = initialization(data_path)
    for qubit in qubits:
        qubit_dir = qubit_path(base_dir, qubit)
        assert os.path.exists(qubit_dir)
        for file in FILES_NAME:
            assert os.path.exists(qubit_dir / file)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="no tensorflow-io-0.32.0's wheel available for Windows",
)
def test_accuracy(data_path):
    table, _, y_test = initialization(data_path)
    real_accuracies = table["accuracy"].tolist()
    # The model is evaluated good if it
    # misclassifys less than two points
    min_accuracy = 1 - 2.0 / len(y_test)

    assert all(acc > min_accuracy for acc in real_accuracies)
