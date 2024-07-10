from dataclasses import dataclass, field
from math import atan2
from pathlib import Path

import numpy as np
import numpy.typing as npt

from qibocal.protocols.utils import cumulative, effective_qubit_temperature


def constructor(_hyperparams):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return QubitFit()


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


normalize = lambda x: x


def dump(model, save_path: Path):
    r"""Dumps the `model` in `save_path`"""
    # relative import to reduce overhead when importing qibocal
    import skops.io as sio

    sio.dump(model, save_path.with_suffix(".skops"))


def predict_from_file(loading_path: Path, input: np.typing.NDArray):
    r"""This function loads the model saved in `loading_path` and returns the
    predictions of `input`."""
    # relative import to reduce overhead when importing qibocal
    import skops.io as sio

    model = sio.load(
        loading_path, trusted=["qibocal.fitting.classifier.qubit_fit.QubitFit"]
    )
    return model.predict(input)


@dataclass
class QubitFit:
    r"""This class deploys a qubit state classifier.

    Args:
        iq_mean0 (np.ndarray): Center of the ground state cloud in the I-Q plane.
        iq_mean1 (np.ndarray): Center of the excited state cloud in the I-Q plane.
        threshold (float): Classifier's threshold.
        angle (float): Rotational angle.
    """  # TODO: add references

    iq_mean0: list = field(default_factory=list)
    iq_mean1: list = field(default_factory=list)
    threshold: float = 0.0
    angle: float = 0.0
    fidelity: float = None
    assignment_fidelity: float = None
    probability_error: float = None
    effective_qubit_temperature: float = None

    def fit(self, iq_coordinates: list, states: list):
        r"""Evaluate the model's parameters given the
        `iq_coordinates` and their relative ``states`
        (reference: <https://arxiv.org/abs/1004.4323>).
        """
        iq_state1 = iq_coordinates[(states == 1)]
        iq_state0 = iq_coordinates[(states == 0)]
        self.iq_mean0 = np.mean(iq_state0, axis=0)
        self.iq_mean1 = np.mean(iq_state1, axis=0)

        vector01 = self.iq_mean1 - self.iq_mean0
        self.angle = -1 * atan2(vector01[1], vector01[0])

        # rotate
        iq_coord_rot = self.rotate(iq_coordinates)

        x_values_state0 = np.sort(iq_coord_rot[(states == 0)][:, 0])
        x_values_state1 = np.sort(iq_coord_rot[(states == 1)][:, 0])

        # evaluate threshold and fidelity
        x_values = np.unique(iq_coord_rot[:, 0])
        cum_distribution_state1 = cumulative(x_values, x_values_state1) / len(
            x_values_state1
        )
        cum_distribution_state0 = cumulative(x_values, x_values_state0) / len(
            x_values_state1
        )

        cum_distribution_diff = np.abs(
            np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
        )
        max_index = np.argmax(cum_distribution_diff)
        self.threshold = x_values[max_index]
        errors_state1 = 1 - cum_distribution_state1[max_index]
        errors_state0 = cum_distribution_state0[max_index]
        self.fidelity = cum_distribution_diff[max_index]
        self.assignment_fidelity = (errors_state1 + errors_state0) / 2
        predictions = self.predict(iq_coordinates)
        self.probability_error = np.sum(np.absolute(states - predictions)) / len(
            predictions
        )

    def effective_temperature(self, predictions, qubit_frequency: float):
        """Calculate effective qubit temperature."""
        prob_1 = np.count_nonzero(predictions) / len(predictions)
        prob_0 = 1 - prob_1
        return effective_qubit_temperature(
            prob_0=prob_0,
            prob_1=prob_1,
            qubit_frequency=qubit_frequency,
            nshots=len(predictions),
        )

    def rotate(self, v):
        c, s = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([[c, -s], [s, c]])
        return v @ rot.T

    def predict(self, inputs: npt.NDArray):
        r"""Classify the `inputs`.

        Returns:
            List of predictions.
        """
        rotated = self.rotate(inputs)
        return (rotated[:, 0] > self.threshold).astype(int)
