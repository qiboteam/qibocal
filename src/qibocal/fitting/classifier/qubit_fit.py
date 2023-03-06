from dataclasses import dataclass
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np

from qibocal.data import DataUnits
from qibocal.fitting.methods import calibrate_qubit_states_fit


@dataclass
class qubit_fit:
    iq_mean0: np.ndarray = np.array([0.0, 0.0])
    iq_mean1: np.ndarray = np.array([0.0, 0.0])
    threshold: float = 0.0
    angle: float = 0.0

    def fit(self, x, y):
        data = raw_to_dataunits(x, y)
        results = calibrate_qubit_states_fit(data, "i[V]", "q[V]", 1, [1]).df
        iq_state0 = results.iloc[0]["average_state0"]
        iq_state1 = results.iloc[0]["average_state1"]
        self.angle = results.iloc[0]["rotation_angle"]
        self.threshold = results.iloc[0]["threshold"]
        self.iq_mean0 = np.array([iq_state0.real, iq_state0.imag])
        self.iq_mean1 = np.array([iq_state1.real, iq_state1.imag])

    def rotate(self, v):
        theta = -1 * self.angle
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(rot, v)

    def translate(self, v):
        return v - self.iq_mean0

    def predict(self, inputs: list[float]):
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


def raw_to_dataunits(x, y):
    options = ["qubit", "state"]
    data = DataUnits(options=options)
    data_dict = {
        "MSR[V]": [0] * len(y),
        "i[V]": x[:, 0].tolist(),
        "q[V]": x[:, 1].tolist(),
        "phase[rad]": [0] * len(y),
        "state": y.tolist(),
        "qubit": [1] * len(y),
    }
    data.load_data_from_dict(data_dict)

    return data


def hyperopt(x_train, y_train, _path):
    return {}


def constructor(_hyperparams):
    return qubit_fit()


def normalize(unormalize):
    return unormalize
