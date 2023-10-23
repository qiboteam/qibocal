from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import qubit_fit as qfit


def constructor(_hyperparams):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return QbloxFit()


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
dump = qfit.dump
predict_from_file = qfit.predict_from_file


@dataclass
class QbloxFit:
    r"""This class deploys the Qblox qubit state classifier described
    in the [documentation](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/conditional_playback.html#Measure-qubit-histogram).

    Args:
        threshold (float): Classifier's threshold.
        angle (float): Rotational angle.

    """

    threshold: float = 0.0
    angle: float = 0.0

    def fit(self, iq_coordinates, states: list):
        state1 = [complex(*i) for i in iq_coordinates[(states == 1)]]
        state0 = [complex(*i) for i in iq_coordinates[(states == 0)]]
        self.angle = np.mod(-np.angle(np.mean(state1) - np.mean(state0)), 2 * np.pi)
        self.threshold = (
            np.exp(1j * self.angle) * (np.mean(state1) + np.mean(state0))
        ).real / 2

    def predict(self, inputs: npt.NDArray):
        inputs = np.array([complex(*i) for i in inputs])
        return ((np.exp(1j * self.angle) * inputs).real > self.threshold).astype(int)
