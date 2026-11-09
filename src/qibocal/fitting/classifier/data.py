import numpy as np
from sklearn.model_selection import train_test_split


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
    states = np.array(qubit_data[["state"]].tolist())[:, 0]
    return train_test_split(
        np.array(qubit_data[["i", "q"]].tolist())[:, :],
        states,
        test_size=test_size,
        random_state=0,
        shuffle=True,
        stratify=states,
    )
