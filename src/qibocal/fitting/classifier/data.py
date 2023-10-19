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
    data0 = data.data[qubit, 0].tolist()
    data1 = data.data[qubit, 1].tolist()
    return train_test_split(
        np.array(np.concatenate((data0, data1))),
        np.array([0] * len(data0) + [1] * len(data1)),
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )
