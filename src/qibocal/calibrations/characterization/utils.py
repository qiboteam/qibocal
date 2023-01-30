import pathlib

import joblib
import numpy as np


def iq_to_prob(i, q, mean_gnd, mean_exc):
    """
    mean_gnd and mean_exc are the mean of the ground and excited states in the
    complex form (i + 1j * q)
    """
    state = i + 1j * q
    state = state - mean_gnd
    mean_exc = mean_exc - mean_gnd
    state = state * np.exp(-1j * np.angle(mean_exc))
    mean_exc = mean_exc * np.exp(-1j * np.angle(mean_exc))
    return np.real(state) / np.real(mean_exc)
