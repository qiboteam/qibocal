# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit


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


def flipping_fit_3D(x_data, y_data):
    pguess = [0.0003, np.mean(y_data), -18, 0]  # epsilon guess parameter
    popt, pcov = curve_fit(flipping, x_data, y_data, p0=pguess)
    return popt


def flipping_fit_2D(x_data, y_data):
    pguess = [0.0003, np.mean(y_data), 18, 0]  # epsilon guess parameter
    popt, pcov = curve_fit(flipping, x_data, y_data, p0=pguess)
    return popt


def flipping(x, p0, p1, p2, p3):
    # A fit to Flipping Qubit oscillation
    # Epsilon                       : p[0]
    # Offset                        : p[1]
    # Period of oscillation         : p[2]
    # phase for the first point corresponding to pi/2 rotation   : p[3]
    return np.sin(x * 2 * np.pi / p2 + p3) * p0 + p1


def classify(point: complex, mean_gnd, mean_exc):
    import math

    """Classify the given state as |0> or |1>."""

    def distance(a, b):
        return math.sqrt(
            (np.real(a) - np.real(b)) ** 2 + (np.imag(a) - np.imag(b)) ** 2
        )

    return int(distance(point, mean_exc) < distance(point, mean_gnd))


def fit_drag_tunning(res1, res2, beta_params):

    # find line of best fit
    a, b = np.polyfit(beta_params, res1, 1)
    c, d = np.polyfit(beta_params, res2, 1)

    # find interception point
    xi = (b - d) / (c - a)
    yi = a * xi + b

    return xi
