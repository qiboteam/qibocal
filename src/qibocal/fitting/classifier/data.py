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
    qubit_data = data.data[qubit]
    return train_test_split(
        np.array(qubit_data[["i", "q"]].tolist())[:, :],
        np.array(qubit_data[["state"]].tolist())[:, 0],
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )
