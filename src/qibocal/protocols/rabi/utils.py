from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from qibolab import Delay, PulseLike, PulseSequence
from qibolab._core.identifier import ChannelId
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    COLORBAND,
    COLORBAND_LINE,
    angle_wrap,
    guess_period,
    table_dict,
    table_html,
)
from qibocal.update import replace

from .parent_classes import RabiData, RabiResults

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
            y=qubit_data.signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=qubit_data.phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            2 * len(rabi_parameters),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )
        pulse_name = "Pi-half pulse" if rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                qubit,
                [f"{pulse_name} amplitude [a.u.]", f"{pulse_name} length [ns]"],
                [np.round(fit.amplitude[qubit], 3), np.round(fit.length[qubit], 3)],
            )
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title=title,
            yaxis_title="Signal [a.u.]",
            xaxis2_title=title,
            yaxis2_title="Phase [rad]",
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
    probs = qubit_data.prob
    error_bars = qubit_data.error
    rabi_parameters = getattr(qubit_data, quantity)
    fig = go.Figure(
        [
            go.Scatter(
                x=rabi_parameters,
                y=qubit_data.prob,
                opacity=1,
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((rabi_parameters, rabi_parameters[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            2 * len(rabi_parameters),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
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
        )

    figures.append(fig)

    return figures, fitting_report


def extract_rabi(data: RabiData) -> tuple[str, str, Callable]:
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


def single_qubit_rabi_sequence(
    target: QubitId,
    drive_line: QubitId,
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
) -> tuple[PulseSequence, PulseLike, ChannelId, dict]:
    """Generate a single qubit Rabi sequence given a specific qubit and the line we want to drive it."""

    single_q_sequence = PulseSequence()
    update = {}
    natives_pulses = platform.natives.single_qubit[target]

    qd_channel, qd_pulse = natives_pulses.RX90()[0] if rx90 else natives_pulses.RX()[0]
    if target != drive_line:
        # used when q is being driven with another line (cross rabi)
        cross_channel = platform.qubits[drive_line].drive
        qubit_freq = platform.parameters.configs[qd_channel].frequency
        update |= {cross_channel: {"frequency": qubit_freq}}
        qd_channel = cross_channel

    if pulse_ampl is not None:
        qd_pulse = replace(qd_pulse, amplitude=pulse_ampl)

    if pulse_duration is not None:
        qd_pulse = replace(qd_pulse, duration=pulse_duration)

    if rx90:
        single_q_sequence.append((qd_channel, qd_pulse))

    single_q_sequence.append((qd_channel, qd_pulse))

    return single_q_sequence, qd_pulse, qd_channel, update


def sequence_amplitude(
    targets: list[QubitId],
    drive_lines: list[QubitId],
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
) -> tuple[PulseSequence, list[PulseLike], dict[QubitId, float], dict]:
    """Generate Rabi pulse sequences for amplitude sweeping on multiple qubits and generic drive lines scheme."""

    sequence = PulseSequence()
    qd_pulses: list[PulseLike] = []
    durations: dict[QubitId, float] = {}
    updates = {}
    for q, d in zip(targets, drive_lines):
        # creating Rabi sequence for a (qubit, drive_line) pair
        single_q_seq, single_q_pulse, _, single_q_update = single_qubit_rabi_sequence(
            target=q,
            drive_line=d,
            platform=platform,
            pulse_duration=pulse_duration,
            pulse_ampl=pulse_ampl,
            rx90=rx90,
        )
        qd_pulses.append(single_q_pulse)
        durations[q] = single_q_pulse.duration
        updates |= single_q_update

        # aligning readout pulses to single qubit sequence
        single_q_seq |= PulseSequence(platform.natives.single_qubit[q].MZ())

        # adding the single qubit sequence to the complete one
        sequence += single_q_seq

    return sequence, qd_pulses, durations, updates


def sequence_length(
    targets: list[QubitId],
    drive_lines: list[QubitId],
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, list[PulseLike], list[Delay], dict[QubitId, float], dict]:
    """Generate Rabi pulse sequences for duration sweeping on multiple qubits and generic drive lines scheme."""

    sequence = PulseSequence()
    amplitudes: dict[QubitId, float] = {}
    updates = {}
    qd_pulses: list[PulseLike] = []
    delays: list[Delay] = []
    for q, d in zip(targets, drive_lines):
        # creating Rabi sequence for a (qubit, drive_line) pair
        single_q_seq, single_q_pulse, single_q_channel, single_q_update = (
            single_qubit_rabi_sequence(
                target=q,
                drive_line=d,
                platform=platform,
                pulse_duration=pulse_duration,
                pulse_ampl=pulse_ampl,
                rx90=rx90,
            )
        )
        sequence += single_q_seq
        qd_pulses.append(single_q_pulse)
        amplitudes[q] = single_q_pulse.amplitude
        updates |= single_q_update

        # appending readout pulses
        ro_channel, ro_pulse = platform.natives.single_qubit[q].MZ()[0]
        if use_align:
            sequence.align([single_q_channel, ro_channel])
        else:
            delays.append(Delay(duration=single_q_pulse.duration))
            sequence.append((ro_channel, delays[-1]))
        sequence.append((ro_channel, ro_pulse))

    return sequence, qd_pulses, delays, amplitudes, updates


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
