from typing import Optional

import numpy as np
from qibolab import Delay, Platform, PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.protocols.utils import GHZ_TO_HZ, fallback_period, guess_period

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
    targets: list[QubitId],
    wait: int = 0,
    target_qubit: Optional[QubitId] = None,
):
    """Pulse sequence used in Ramsey (detuned) experiments.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ
    """
    delays = []
    sequence = PulseSequence()
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]

        qd_channel, qd_pulse = natives.R(theta=np.pi / 2)[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_delay = Delay(duration=wait)
        ro_delay = Delay(duration=wait)

        sequence.extend(
            [
                (qd_channel, qd_pulse),
                (qd_channel, qd_delay),
                (qd_channel, qd_pulse),
                (ro_channel, Delay(duration=2 * qd_pulse.duration)),
                (ro_channel, ro_delay),
                (ro_channel, ro_pulse),
            ]
        )

        delays.extend([qd_delay, ro_delay])
        if target_qubit is not None:
            assert target_qubit not in targets, (
                f"Cannot run Ramsey experiment on qubit {target_qubit} if it is already in Ramsey sequence."
            )
            natives = platform.natives.single_qubit[target_qubit]
            sequence += natives.RX()

    return sequence, delays


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


def process_fit(
    popt: list[float], perr: list[float], qubit_frequency: float, detuning: float
):
    """Processing Ramsey fitting results."""

    delta_fitting = popt[2] / (2 * np.pi)
    if detuning is not None:
        sign = np.sign(detuning)
        delta_phys = int(sign * (delta_fitting * GHZ_TO_HZ - np.abs(detuning)))
    else:
        delta_phys = int(delta_fitting * GHZ_TO_HZ)

    corrected_qubit_frequency = int(qubit_frequency - delta_phys)
    t2 = 1 / popt[4]
    new_frequency = [
        corrected_qubit_frequency,
        perr[2] * GHZ_TO_HZ / (2 * np.pi),
    ]
    t2 = [t2, perr[4] * (t2**2)]

    delta_phys_measure = [
        -delta_phys,
        perr[2] * GHZ_TO_HZ / (2 * np.pi),
    ]
    delta_fitting_measure = [
        -delta_fitting * GHZ_TO_HZ,
        perr[2] * GHZ_TO_HZ / (2 * np.pi),
    ]

    return new_frequency, t2, delta_phys_measure, delta_fitting_measure, popt
