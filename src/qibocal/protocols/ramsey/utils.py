from typing import Optional

import numpy as np
from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.protocols.utils import GHZ_TO_HZ, fallback_period, guess_period

POPT_EXCEPTION = [0, 0, 0, 0, 1]
"""Fit parameters output to handle exceptions"""
PERR_EXCEPTION = [1] * 5
"""Fit errors to handle exceptions; their choice has no physical meaning
and is meant to avoid breaking the code."""
THRESHOLD = 0.5
"""Threshold parameters for find_peaks to guess frequency for sinusoidal fit."""
DAMPED_CONSTANT = 1.5
"""Scaling factor to recover amplitude from quantiles.

Measuring intermediate quantiles is less noise sensitive then meauring extremal points
(minimum and maximum), but it is not a direct measurement of the amplitude itself.
For pure sinusoidal oscillations, the scaling from the value associated to a given
quantile and the amplitude is asymptotically fixed, for a large number of oscillations.
Assuming that samples are dense enough that they could be represented by the continuous
distribution, essentially projecting a uniform measure over an interval through a single
sinusoidal oscillation.

.. todo::

    Move the above paragraph to the Rabi equivalent, and replace here with a reference

However, for damped oscillations, the factor is not easily determined, since the
value associated to a certian quantile depends on the observation window extent, and the
ratio between the decay rate and the oscillation.

Assuming a mild decay, and we can approximate this factor with the same one for the
pure oscillation. This can be assumed to be slightly decreased because of the dampening,
but there is no general control about how much.
By reducing the amplitude by this rough 30%, the estimation will lend closer to the
actual amplitude. We rely anyhow on the fit to determine the precise value.
"""


def ramsey_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
    target_qubit: Optional[QubitId] = None,
    flux_pulse_amplitude: Optional[float] = None,
):
    """Pulse sequence used in Ramsey (detuned) experiments.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ
    """
    delays = 2 * len(targets) * [Delay(duration=wait)]
    if flux_pulse_amplitude is not None:
        flux_pulses = len(targets) * [
            Pulse(duration=0, amplitude=flux_pulse_amplitude, envelope=Rectangular())
        ]
    else:
        flux_pulses = []
    sequence = PulseSequence()
    for i, qubit in enumerate(targets):
        natives = platform.natives.single_qubit[qubit]
        qd_channel = platform.qubits[qubit].drive
        rx90_sequence = natives.R(theta=np.pi / 2)
        ro_channel, ro_pulse = natives.MZ()[0]

        sequence += rx90_sequence
        sequence.append((qd_channel, delays[2 * i]))
        sequence += rx90_sequence
        sequence.extend(
            [
                (ro_channel, Delay(duration=2 * rx90_sequence.duration)),
                (ro_channel, delays[2 * i + 1]),
                (ro_channel, ro_pulse),
            ]
        )
        if flux_pulse_amplitude is not None:
            flux_channel = platform.qubits[qubit].flux
            sequence.append((flux_channel, Delay(duration=rx90_sequence.duration)))
            sequence.append((flux_channel, flux_pulses[i]))
        if target_qubit is not None:
            assert target_qubit not in targets, (
                f"Cannot run Ramsey experiment on qubit {target_qubit} if it is already in Ramsey sequence."
            )
            natives = platform.natives.single_qubit[target_qubit]
            sequence += natives.RX()

    return sequence, delays + flux_pulses


def ramsey_fit(x, offset, amplitude, delta, phase, decay):
    """Dumped sinusoidal fit."""
    return offset + amplitude * np.sin(x * delta + phase) * np.exp(-x * decay)


def angle_wrap(angle: float):
    """Wrap an angle from [-np.inf,np.inf] into the [0,2*np.pi] domain"""
    return angle % (2 * np.pi)


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
    median_sig = np.median(y)
    q80 = np.quantile(y, 0.8)
    q20 = np.quantile(y, 0.2)
    amplitude_guess = abs(q80 - q20) / DAMPED_CONSTANT

    p0 = [
        median_sig,
        amplitude_guess,
        omega,
        np.pi / 2,  # since at tau=0 the probability of the excited state is maximum
        1,
    ]

    popt, perr = curve_fit(
        ramsey_fit,
        x,
        y,
        p0=p0,
        maxfev=5000,
        bounds=(
            [0, 0, 0, -np.inf, 0],
            [1, 1, np.inf, np.inf, np.inf],
        ),
        sigma=err,
    )
    popt = [
        delta_y * popt[0] + y_min,
        delta_y * popt[1] * np.exp(x_min * popt[4] / delta_x),
        popt[2] / delta_x,
        angle_wrap(popt[3] - x_min * popt[2] / delta_x),
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
