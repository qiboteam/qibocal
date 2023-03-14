from dataclasses import dataclass
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np

from ...data import DataUnits
from ..methods import calibrate_qubit_states_fit


@dataclass
class qubit_fit:
    r"""This class deploys a qubit state classifier.

    Args:
        iq_mean0 (np.ndarray): Center of the ground state cloud in the I-Q plane.
        iq_mean1 (np.ndarray): Center of the excited state cloud in the I-Q plane.
        threshold (float): Classifier's threshold.
        angle (float): Rotational angle.
    """  # TODO: add references

    iq_mean0: np.ndarray = np.array([0.0, 0.0])
    iq_mean1: np.ndarray = np.array([0.0, 0.0])
    threshold: float = 0.0
    angle: float = 0.0

    def fit(self, x, y):
        r"""Evaluate the model's parameters given the
        input data (`x`,`y`).
        """
        data = _raw_to_dataunits(x, y)
        results = calibrate_qubit_states_fit(data, "i[V]", "q[V]", 1, [1]).df
        iq_state0 = results.iloc[0]["average_state0"]  # pylint: disable=maybe-no-member
        iq_state1 = results.iloc[0]["average_state1"]  # pylint: disable=maybe-no-member
        self.angle = results.iloc[0][
            "rotation_angle"
        ]  # pylint: disable=maybe-no-member
        self.threshold = results.iloc[0]["threshold"]  # pylint: disable=maybe-no-member
        self.iq_mean0 = np.array([iq_state0.real, iq_state0.imag])
        self.iq_mean1 = np.array([iq_state1.real, iq_state1.imag])

    def rotate(self, v):
        theta = -1 * self.angle
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(rot, v)

    def translate(self, v):
        return v - self.iq_mean0

    def predict(self, inputs: list[float]):
        r"""Classify the `inputs`.

        Returns:
            List of predictions.
        """
        predictions = []

        for input in inputs:
            input = np.array(input)

            input = self.translate(input)
            input = self.rotate(input)

            if input[0] < self.threshold:
                predictions.append(0.0)

            else:
                predictions.append(1.0)

        return predictions


def _raw_to_dataunits(iq_couples, states):
    r"""Return a `DataUnits` that stores the data contained in `iq_couples` and `states`."""
    options = ["qubit", "state"]
    length = len(states)
    data = DataUnits(options=options)
    data_dict = {
        "MSR[V]": [0] * length,
        "i[V]": iq_couples[:, 0].tolist(),
        "q[V]": iq_couples[:, 1].tolist(),
        "phase[rad]": [0] * length,
        "state": states.tolist(),
        "qubit": [1] * length,
    }
    data.load_data_from_dict(data_dict)

    return data


def hyperopt(_x_train, _y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    return {}


def constructor(_hyperparams):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return qubit_fit()


def normalize(unormalize):
    r"""Return a model that implement a step of data normalisation.

    Args:
        unormalize: Model.
    """
    return unormalize
