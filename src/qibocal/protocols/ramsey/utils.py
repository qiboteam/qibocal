from typing import Optional

import numpy as np
from qibolab import Platform
from qibolab.identifier import QubitId
from qibolab.sequence import PulseSequence
from scipy.optimize import curve_fit

from qibocal.protocols.utils import fallback_period, guess_period

POPT_EXCEPTION = [0, 0, 0, 0, 1]
"""Fit parameters output to handle exceptions"""
PERR_EXCEPTION = [1] * 5
"""Fit errors to handle exceptions; their choice has no physical meaning
and is meant to avoid breaking the code."""
THRESHOLD = 0.5
"""Threshold parameters for find_peaks to guess
   frequency for sinusoidal fit."""


def ramsey_sequence(
    platform: Platform,
    qubit: QubitId,
    wait: Optional[int] = 0,
    detuning: Optional[int] = 0,
):
    """Pulse sequence used in Ramsey (detuned) experiments.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ

    If detuning is specified the RX90 pulses will be sent to
    frequency = drive_frequency + detuning
    """

    sequence = PulseSequence()
    first_pi_half_pulse = platform.create_RX90_pulse(qubit, start=0)
    second_pi_half_pulse = platform.create_RX90_pulse(
        qubit, start=first_pi_half_pulse.finish + wait
    )

    # apply detuning:
    first_pi_half_pulse.frequency += detuning
    second_pi_half_pulse.frequency += detuning
    readout_pulse = platform.create_qubit_readout_pulse(
        qubit, start=second_pi_half_pulse.finish
    )

    sequence.add(first_pi_half_pulse, second_pi_half_pulse, readout_pulse)
    return sequence


def ramsey_fit(x, offset, amplitude, delta, phase, decay):
    """Dumped sinusoidal fit."""
    return offset + amplitude * np.sin(x * delta + phase) * np.exp(-x * decay)


def fitting(x: list, y: list, errors: list = None) -> list:
    """
    Given the inputs list `x` and outputs one `y`, this function fits the
    `ramsey_fit` function and returns a list with the fit parameters.
    """
    y_max = np.max(y)
    y_min = np.min(y)
    x_max = np.max(x)
    x_min = np.min(x)
    delta_y = y_max - y_min
    delta_x = x_max - x_min
    y = (y - y_min) / delta_y
    x = (x - x_min) / delta_x
    err = errors / delta_y if errors is not None else None

    period = fallback_period(guess_period(x, y))
    omega = 2 * np.pi / period
    p0 = [
        0.5,
        0.5,
        omega,
        0,
        1,
    ]
    popt, perr = curve_fit(
        ramsey_fit,
        x,
        y,
        p0=p0,
        maxfev=5000,
        bounds=(
            [0, 0, 0, -np.pi, 0],
            [1, 1, np.inf, np.pi, np.inf],
        ),
        sigma=err,
    )
    popt = [
        delta_y * popt[0] + y_min,
        delta_y * popt[1] * np.exp(x_min * popt[4] / delta_x),
        popt[2] / delta_x,
        popt[3] - x_min * popt[2] / delta_x,
        popt[4] / delta_x,
    ]
    perr = np.sqrt(np.diag(perr))
    perr = [
        delta_y * perr[0],
        delta_y
        * np.exp(x_min * popt[4] / delta_x)
        * np.sqrt(perr[1] ** 2 + (popt[1] * x_min * perr[4] / delta_x) ** 2),
        perr[2] / delta_x,
        np.sqrt(perr[3] ** 2 + (perr[2] * x_min / delta_x) ** 2),
        perr[4] / delta_x,
    ]
    return popt, perr
