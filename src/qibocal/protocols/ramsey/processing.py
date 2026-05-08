"""RAMSEY protocol processing utilities.

Contains fitting routines, result transformations, and figure generation
helpers for Ramsey experiments.
"""

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    GHZ_TO_HZ,
    angle_wrap,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

from .acquisition import RamseyResults

POPT_EXCEPTION = [0, 0, 0, 0, 1]
"""Fit parameters output to handle exceptions"""
PERR_EXCEPTION = [1] * 5
"""Fit errors to handle exceptions; their choice has no physical meaning
and is meant to avoid breaking the code."""
THRESHOLD = 0.5
"""Threshold parameters for find_peaks to guess frequency for sinusoidal fit."""
DAMPED_CONSTANT = 1.5
"""See :const:`rabi.utils.QUANTILE_CONSTANT` for details.

In general in Ramsey it's intended to observe the decay of the signal due to decoherence, hence we
need to correct and decrease a little the value of :const:`rabi.utils.DAMPED_CONSTANT`;
Indeed, for damped oscillations, the factor is not easily determined, since the
value associated to a certian quantile depends on the observation window extent, and the
ratio between the decay rate and the oscillation.

Assuming a mild decay, and we can approximate this factor with the same one for the
pure oscillation. This can be assumed to be slightly decreased because of the dampening,
but there is no general control about how much.
By reducing the amplitude by this rough 30%, the estimation will lend closer to the
actual amplitude. We rely anyhow on the fit to determine the precise value.
"""


def ramsey_update(
    results: RamseyResults, platform: CalibrationPlatform, target: QubitId
) -> None:
    """Update the platform calibration with the results of the Ramsey experiment."""
    if results.detuning is not None:
        update.drive_frequency(results.frequency[target][0], platform, target)
        platform.calibration.single_qubits[
            target
        ].qubit.frequency_01 = results.frequency[target][0]
    else:
        update.t2(results.t2[target], platform, target)


def ramsey_fit(x, offset, amplitude, delta, phase, decay) -> NDArray | float:
    """Dumped sinusoidal fit."""
    return offset + amplitude * np.sin(x * delta + phase) * np.exp(-x * decay)


def fitting(x: list, y: list) -> tuple[list[float], list[float]]:
    """
    Given the inputs list `x` and outputs one `y`, this function fits the
    `ramsey_fit` function and returns a list with the fit parameters.
    """

    # performing a min-max scaling on x and y arrays
    y_max = np.max(y)
    y_min = np.min(y)
    x_max = np.max(x)
    x_min = np.min(x)
    delta_y = y_max - y_min
    delta_x = x_max - x_min
    y = (y - y_min) / delta_y
    x = (x - x_min) / delta_x

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
    )

    # inverting the scaling
    popt = [
        delta_y * popt[0] + y_min,
        delta_y * popt[1] * np.exp(x_min * popt[4] / delta_x),
        popt[2] / delta_x,
        angle_wrap(popt[3] - x_min * popt[2] / delta_x),
        popt[4] / delta_x,
    ]

    perr = np.sqrt(np.diag(perr))
    # error propagation in the original units
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
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
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


def fit_plot(
    target: QubitId, fit: RamseyResults, waits: NDArray, fig: go.Figure
) -> str:
    """Generate the fit trace and summary table for Ramsey data."""

    fit_waits = np.linspace(min(waits), max(waits), 20 * len(waits))
    fig.add_trace(
        go.Scatter(
            x=waits,
            y=ramsey_fit(fit_waits, *fit.fitted_parameters[target]),
            name="Fit",
            mode="lines",
        )
    )

    return table_html(
        table_dict(
            target,
            [
                "Delta Frequency [Hz]",
                "Delta Frequency (with detuning) [Hz]",
                "Drive Frequency [Hz]",
                "T2* [ns]",
            ],
            [
                fit.delta_phys[target],
                fit.delta_fitting[target],
                fit.frequency[target],
                fit.t2[target],
            ],
            display_error=True,
        )
    )


def signal_plot(
    waits: NDArray,
    signal: NDArray,
    target: QubitId,
    fit: RamseyResults | None,
    yaxis_title: str,
) -> tuple[list[go.Figure], str]:
    """Create a signal scatter plot and optional fit report."""

    fitting_report = ""
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=signal,
                opacity=1,
                name=yaxis_title,
                showlegend=True,
                legendgroup=yaxis_title,
                mode="markers",
            ),
        ]
    )

    if fit is not None:
        fitting_report = fit_plot(
            target=target,
            fit=fit,
            waits=waits,
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title=yaxis_title,
    )

    return [fig], fitting_report
