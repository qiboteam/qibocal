import numpy as np
from scipy.optimize import curve_fit

from qibocal.protocols.utils import angle_wrap, guess_period

QUANTILE_CONSTANT_RABI = 1.5
"""Scaling factor to recover signal amplitude from quantiles.

Measuring intermediate quantiles is less noise sensitive then measuring extremal points
(minimum and maximum), but it is not a direct measurement of the amplitude itself.
For pure sinusoidal oscillations, the scaling from the value associated to a given
quantile and the amplitude is asymptotically fixed, for a large number of oscillations.
Assuming that samples are dense enough that they could be represented by the continuous
distribution, essentially projecting a uniform measure over an interval through a single
sinusoidal oscillation.
"""


def rabi_amplitude_function(x, offset, amplitude, period, phase):
    """
    Fit function of Rabi amplitude signal experiment.

    Args:
        x: Input data.
    """
    return offset + amplitude * np.cos(2 * np.pi * x / period + phase)


def rabi_length_function(x, offset, amplitude, period, phase, t2_inv):
    """
    Fit function of Rabi length signal experiment.

    Args:
        x: Input data.
    """
    return offset + amplitude * np.cos(2 * np.pi * x / period + phase) * np.exp(
        -x * t2_inv
    )


def rabi_initial_guess(x, y, experiment: str, signal: bool):
    period = guess_period(x, y)
    median_sig = np.median(y)
    q80 = np.quantile(y, 0.8)
    q20 = np.quantile(y, 0.2)
    amplitude_guess = abs(q80 - q20) / QUANTILE_CONSTANT_RABI
    phase_guess = np.pi if not signal else np.pi / 2

    if experiment == "length":
        return [median_sig, amplitude_guess, period, phase_guess, 0]
    else:
        return [median_sig, amplitude_guess, period, phase_guess]


def extract_rabi(data):
    """
    Extract Rabi fit info.
    """
    if "RabiAmplitude" in data.__class__.__name__:
        return "amp", "Amplitude [dimensionless]", rabi_amplitude_function
    if "RabiLength" in data.__class__.__name__:
        return "length", "Time [ns]", rabi_length_function
    raise RuntimeError("Data has to be a data structure of the Rabi routines.")


def period_correction_factor(phase: float):
    r"""Correct period by taking phase into account.

    https://github.com/qiboteam/qibocal/issues/656

    We want to find the first maximum or minimum which will
    correspond to an exchange of population between 0 and 1.
    To find it we need to solve the following equation
    :math:`\cos(2 \pi x / T + \phi) = \pm 1` which will give us
    the following solution

    .. math::

        x = ( k - \phi / \pi) T / 2


    for integer :math:`k`, which is chosen such that we get the smallest
    multiplicative correction compared to :math:`T/2`.

    """

    x = phase / np.pi
    return np.round(1 + x) - x


def fit_length_function(
    x, y, guess, sigma=None, signal=True, x_limits=(None, None), y_limits=(None, None)
) -> tuple[list[float], list[float], float]:
    popt, perr = curve_fit(
        rabi_length_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            [0, -1 if signal else 0, 0, -np.inf, 0],
            [1, 1, np.inf, np.inf, np.inf],
        ),
        sigma=sigma,
    )
    x_min = x_limits[0]
    x_max = x_limits[1]
    y_min = y_limits[0]
    y_max = y_limits[1]
    if signal is False:
        popt = [
            popt[0],
            popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
            popt[2] * (x_max - x_min),
            angle_wrap(popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min)),
            popt[4] / (x_max - x_min),
        ]
        perr = np.sqrt(np.diag(perr))
    else:
        popt = [  # change it according to the fit function
            (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
            (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
            popt[2] * (x_max - x_min),
            popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
            popt[4] / (x_max - x_min),
        ]

    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])
    return popt, perr.tolist(), pi_pulse_parameter


def fit_amplitude_function(
    x, y, guess, sigma=None, signal=True, x_limits=(None, None), y_limits=(None, None)
) -> tuple[list[float], list[float], float]:
    popt, perr = curve_fit(
        rabi_amplitude_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            [0, 0, 0, -np.inf],
            [1, 1, np.inf, np.inf],
        ),
        sigma=sigma,
    )
    if signal is False:
        perr = np.sqrt(np.diag(perr))
    if None not in y_limits and None not in x_limits:
        popt = [
            y_limits[0] + (y_limits[1] - y_limits[0]) * popt[0],
            (y_limits[1] - y_limits[0]) * popt[1],
            popt[2] * (x_limits[1] - x_limits[0]),
            angle_wrap(
                popt[3]
                - 2 * np.pi * x_limits[0] / (x_limits[1] - x_limits[0]) / popt[2]
            ),
        ]
    else:
        popt = popt.tolist()

    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])

    return popt, perr.tolist(), pi_pulse_parameter
