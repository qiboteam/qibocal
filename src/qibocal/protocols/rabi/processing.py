from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    angle_wrap,
    guess_period,
    table_dict,
    table_html,
)
from qibocal.result import collect, magnitude, phase

from .parent_classes import RabiData, RabiFreqResults, RabiResults

FIT_COLOUR_LINE = "rgba(255, 57, 57, 0.8)"
"""Rgba code of the colour used for plotting the fit."""
QUANTILE_CONSTANT = 1.5
"""Scaling factor to recover signal amplitude from quantiles.

Measuring intermediate quantiles is less noise sensitive then measuring extremal points
(minimum and maximum), but it is not a direct measurement of the amplitude itself.
For pure sinusoidal oscillations, the scaling from the value associated to a given
quantile and the amplitude is asymptotically fixed, for a large number of oscillations.
Assuming that samples are dense enough that they could be represented by the continuous
distribution, essentially projecting a uniform measure over an interval through a single
sinusoidal oscillation.
"""


def update_rabi_parameters(
    results: RabiResults | RabiFreqResults,
    platform: CalibrationPlatform,
    target: QubitId,
) -> None:
    """Updating RX or RX90 parameters if the drive line is the physical line for qubit `target`"""
    drive_line = results.drive_lines[target]
    if target == drive_line:
        update.drive_duration(results.length[target], results.rx90, platform, target)
        update.drive_amplitude(
            results.amplitude[target], results.rx90, platform, target
        )
        if isinstance(results, RabiFreqResults):
            update.drive_frequency(results.frequency[target], platform, target)
    else:
        cross_rabi_freq = results.length[target][0] * results.amplitude[target][0]
        _, rx_pulse = platform.parameters.native_gates.single_qubit[target].RX()[0]
        rx_rabi_freq = rx_pulse.amplitude * rx_pulse.duration

        # taking the ratio between the direct RX pulse and the cross one
        # from https://arxiv.org/pdf/2112.03708
        # here the first index is the qubit I want to drive and the second is the
        # drive line I want to pulse form.
        platform.calibration.microwave_crosstalk_matrix[target, drive_line] = (
            cross_rabi_freq / rx_rabi_freq
        )


def rabi_amplitude_function(
    x: ArrayLike,
    offset: float,
    amplitude: float,
    period: float,
    phase: float,
) -> NDArray:
    """
    Fit function of Rabi amplitude signal experiment.
    """
    return offset + amplitude * np.cos(2 * np.pi * x / period + phase)


def rabi_length_function(
    x: ArrayLike,
    offset: float,
    amplitude: float,
    period: float,
    phase: float,
    t2_inv: float,
) -> NDArray:
    """
    Fit function of Rabi length signal experiment.
    """
    return offset + amplitude * np.cos(2 * np.pi * x / period + phase) * np.exp(
        -x * t2_inv
    )


def rabi_initial_guess(
    x: ArrayLike, y: ArrayLike, experiment: str, signal: bool
) -> list[float]:
    """Generate an initial guess for Rabi curve fitting.

    For a length experiment, the returned list contains [offset, amplitude, period,
    phase, t2_inv]; otherwise it contains [offset, amplitude, period, phase].
    """
    period = guess_period(x, y)
    median_sig = np.median(y)
    q80 = np.quantile(y, 0.8)
    q20 = np.quantile(y, 0.2)
    amplitude_guess = abs(q80 - q20) / QUANTILE_CONSTANT
    phase_guess = np.pi if not signal else np.pi / 2

    if experiment == "length":
        return [median_sig, amplitude_guess, period, phase_guess, 0]
    else:
        return [median_sig, amplitude_guess, period, phase_guess]


def plot_signal(
    data: RabiData, qubit: QubitId, fit: RabiResults | None, rx90: bool
) -> tuple[list[go.Figure], str]:
    """Create plots for a Rabi experiment signal and phase."""
    quantity, title, fitting = extract_rabi(data)
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Signal [a.u.]",
            "phase [rad]",
        ),
    )

    qubit_data = data[qubit]

    rabi_parameters = getattr(qubit_data, quantity)
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=magnitude(collect(i=qubit_data.i, q=qubit_data.q)),
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
            mode="markers",
            marker=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=phase(collect(i=qubit_data.i, q=qubit_data.q)),
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
            mode="markers",
            marker=dict(color="red"),
        ),
        row=1,
        col=2,
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            50 * len(rabi_parameters),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                mode="lines",
                line=dict(color=FIT_COLOUR_LINE),
            ),
            row=1,
            col=1,
        )
        pulse_name = "Pi-half pulse" if rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                qubit,
                [f"{pulse_name} amplitude [a.u.]", f"{pulse_name} length [ns]"],
                [
                    np.round(fit.amplitude[qubit][0], 3),
                    np.round(fit.length[qubit][0], 3),
                ],
            )
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title=title,
            yaxis_title="Signal [a.u.]",
            xaxis2_title=title,
            yaxis2_title="Phase [rad]",
            title=(
                f"Rabi experiment for qubit {qubit} with "
                + f"drive line {fit.drive_lines[qubit]}"
            ),
        )

    figures.append(fig)

    return figures, fitting_report


def plot_probabilities(
    data: RabiData, qubit: QubitId, fit: RabiResults | None, rx90: bool
) -> tuple[list[go.Figure], str]:
    """
    Generate probability plot for Rabi experiment.
    """
    quantity, title, fitting = extract_rabi(data)
    figures: list[go.Figure] = []
    fitting_report = ""

    qubit_data = data[qubit]
    rabi_parameters = getattr(qubit_data, quantity)
    fig = go.Figure(
        [
            go.Scatter(
                x=rabi_parameters,
                y=qubit_data.prob,
                error_y=dict(
                    type="data",
                    array=qubit_data.error,
                    visible=True,
                    color="red",
                ),
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
                mode="markers",
                marker=dict(color="red"),
            ),
        ]
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            50 * len(rabi_parameters),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                mode="lines",
                line=dict(color=FIT_COLOUR_LINE),
            ),
        )
        pulse_name = "Pi-half pulse" if rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    f"{pulse_name} amplitude [a.u.]",
                    f"{pulse_name} length [ns]",
                    "chi2 reduced",
                ],
                [fit.amplitude[qubit], fit.length[qubit], fit.chi2[qubit]],
                display_error=True,
            )
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title=title,
            yaxis_title="Excited state probability",
            title=(
                f"Rabi experiment for qubit {qubit} with "
                + f"drive line {fit.drive_lines[qubit]}"
            ),
        )

    figures.append(fig)

    return figures, fitting_report


def extract_rabi(data: RabiData) -> tuple[str, str, Callable]:
    """
    Extract Rabi fit info.
    """
    if any([s in data.__class__.__name__ for s in ["RabiAmplitude", "RabiEF"]]):
        return "amp", "Amplitude [a.u.]", rabi_amplitude_function
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
    """Fit Rabi length function to experimental data.

    Performs curve fitting on Rabi oscillation data as a function of pulse duration,
    with exponential decay correction. Rescales fitted parameters based on signal type
    and provided axis limits.
    """

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
        # errors are not propagated correctly
        perr = np.sqrt(np.diag(perr))
    else:
        popt = [  # change it according to the fit function
            (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
            (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
            popt[2] * (x_max - x_min),
            popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
            popt[4] / (x_max - x_min),
        ]
        # errors are not propagated correctly

    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])
    return popt, perr.tolist(), pi_pulse_parameter


def fit_amplitude_function(
    x, y, guess, sigma=None, signal=True, x_limits=(None, None), y_limits=(None, None)
) -> tuple[list[float], list[float], float]:
    """Fit Rabi amplitude function to experimental data.

    Performs curve fitting on Rabi oscillation data as a function of pulse amplitude.
    Rescales fitted parameters based on provided axis limits.
    """
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
