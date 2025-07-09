import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Delay, Platform, PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, QubitId
from qibocal.update import replace

from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html


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


def plot(data, qubit, fit, rx90):
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


def plot_probabilities(data, qubit, fit, rx90):
    quantity, title, fitting = extract_rabi(data)
    figures = []
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


def sequence_amplitude(
    targets: list[QubitId],
    params: Parameters,
    platform: Platform,
    rx90: bool,
) -> tuple[PulseSequence, dict, dict, dict]:
    """Return sequence for rabi amplitude."""

    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]

        qd_channel, qd_pulse = natives.RX90()[0] if rx90 else natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if params.pulse_length is not None:
            qd_pulse = replace(qd_pulse, duration=params.pulse_length)

        durations[q] = qd_pulse.duration
        qd_pulses[q] = qd_pulse
        ro_pulses[q] = ro_pulse

        if rx90:
            sequence.append((qd_channel, qd_pulses[q]))

        sequence.append((qd_channel, qd_pulses[q]))
        sequence.append((ro_channel, Delay(duration=durations[q])))
        sequence.append((ro_channel, ro_pulse))
    return sequence, qd_pulses, ro_pulses, durations


def sequence_length(
    targets: list[QubitId],
    params: Parameters,
    platform: Platform,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, dict, dict, dict]:
    """Return sequence for rabi length."""

    sequence = PulseSequence()
    qd_pulses = {}
    delays = {}
    ro_pulses = {}
    amplitudes = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]

        qd_channel, qd_pulse = natives.RX90()[0] if rx90 else natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if params.pulse_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.pulse_amplitude)

        amplitudes[q] = qd_pulse.amplitude
        ro_pulses[q] = ro_pulse
        qd_pulses[q] = qd_pulse

        if rx90:
            sequence.append((qd_channel, qd_pulse))

        sequence.append((qd_channel, qd_pulse))
        if use_align:
            sequence.align([qd_channel, ro_channel])
        else:
            delays[q] = Delay(duration=16)
            sequence.append((ro_channel, delays[q]))
        sequence.append((ro_channel, ro_pulse))

    return sequence, qd_pulses, delays, ro_pulses, amplitudes


def fit_length_function(
    x, y, guess, sigma=None, signal=True, x_limits=(None, None), y_limits=(None, None)
):
    inf_bounds = [0, -1, 0, -np.pi, 0] if signal else [0, 0, 0, -np.pi, 0]
    popt, perr = curve_fit(
        rabi_length_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            inf_bounds,
            [1, 1, np.inf, np.pi, np.inf],
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
            popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
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
    return popt, perr, pi_pulse_parameter


def fit_amplitude_function(
    x, y, guess, sigma=None, signal=True, x_limits=(None, None), y_limits=(None, None)
):
    popt, perr = curve_fit(
        rabi_amplitude_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            [0, 0, 0, 0],
            [1, 1, np.inf, 2 * np.pi],
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
            popt[3] - 2 * np.pi * x_limits[0] / (x_limits[1] - x_limits[0]) / popt[2],
        ]
    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])
    return popt, perr, pi_pulse_parameter
