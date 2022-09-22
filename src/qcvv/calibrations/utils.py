# -*- coding: utf-8 -*-
import numpy as np


def variable_resolution_scanrange(
    lowres_width, lowres_step, highres_width, highres_step
):
    """Helper function for sweeps."""
    return np.concatenate(
        (
            np.arange(-lowres_width, -highres_width, lowres_step),
            np.arange(-highres_width, highres_width, highres_step),
            np.arange(highres_width, lowres_width, lowres_step),
        )
    )


def classify(point: complex, mean_gnd, mean_exc):
    """Classify the given state as |0> or |1>."""

    def distance(a, b):
        return np.sqrt((np.real(a) - np.real(b)) ** 2 + (np.imag(a) - np.imag(b)) ** 2)

    return int(distance(point, mean_exc) < distance(point, mean_gnd))
