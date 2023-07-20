import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_qubit(data_path: pathlib.Path, qubit):
    r"""Load the information of the qubit `qubit`
    stored in `data_path`.
    Args:
        data_path (path): Data path.
        qubit: Qubit ID.
    Returns:
        Pandas.DataFrame with `qubit`'s data.
    """
    data = pd.read_csv(data_path, skiprows=[1])
    data = data[data.qubit == qubit]
    return data


def generate_models(data, qubit, test_size=0.25):
    r"""Extract from data the values stored
    in the keys `i` and `q` and split them in training and test sets.

    Args:
        data (DataFrame): Qubit's info.
        test_size (float | Optional): The proportion of the dataset to include in the train split.

    Returns:

        - x_train: Training inputs.
        - y_train: Training outputs.
        - x_test: Test inputs.
        - y_test: Test outputs.
    """
    qubit_data0 = data.data[qubit, 0]
    qubit_data1 = data.data[qubit, 1]
    i_shots = np.concatenate((qubit_data0.i, qubit_data1.i))
    q_shots = np.concatenate((qubit_data0.q, qubit_data1.q))
    states = np.concatenate(([0] * len(qubit_data0), [1] * len(qubit_data1)))

    return train_test_split(
        np.column_stack((i_shots, q_shots)),
        states,
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )
