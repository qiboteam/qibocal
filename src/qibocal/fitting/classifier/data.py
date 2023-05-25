import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

COLORS = ["#FF0000", "#0000FF"]
DATAFILE = "data_processing.pdf"


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


def generate_models(data, test_size=0.25):
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
    data["i"] = data["i"].pint.magnitude
    data["q"] = data["q"].pint.magnitude
    input_data = data[["i", "q"]].to_numpy()
    output_data = data["state"].values
    return train_test_split(
        input_data, output_data, test_size=test_size, random_state=0, shuffle=True
    )
