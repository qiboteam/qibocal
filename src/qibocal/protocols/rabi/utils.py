import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Delay, Platform, PulseSequence
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from qibocal.auto.operation import Parameters, QubitId
from qibocal.protocols.utils import (
    COLORBAND,
    COLORBAND_LINE,
    guess_period,
    plot_iq_pca,
    table_dict,
    table_html,
)
from qibocal.result import collect
from qibocal.update import replace

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


def rabi_initial_guess(x, y, experiment: str, signal: bool, axis: int = -1):
    period = guess_period(x, y, axis=axis)
    median_sig = np.median(y, axis=axis)
    q80 = np.quantile(y, 0.8, axis=axis)
    q20 = np.quantile(y, 0.2, axis=axis)
    amplitude_guess = np.abs(q80 - q20) / QUANTILE_CONSTANT_RABI

    phase_guess = np.pi if not signal else np.pi / 2
    zeros = 0
    if not np.isscalar(period):
        zeros = np.zeros_like(period)
        phase_guess = np.full_like(period, phase_guess)

    if experiment == "length":
        return [median_sig, amplitude_guess, period, phase_guess, zeros]
    else:
        return [median_sig, amplitude_guess, period, phase_guess]


def plot(data, qubit, fit, rx90):
    quantity, title, fitting = extract_rabi(data)
    fitting_report = ""

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.15,
        subplot_titles=(
            "IQ Plane",
            "Principal Axis",
            "Second Axis",
        ),
        row_heights=[0.5, 0.35, 0.15],
    )

    qubit_data = data[qubit]
    quadratures = collect(qubit_data.i, qubit_data.q)

    # initialize a PCA instance and fit it to the quadrature data
    pca = PCA().fit(quadratures)
    # apply the pca rotation to the iq signal
    pca_signal = pca.transform(quadratures)

    rabi_parameters = getattr(qubit_data, quantity)

    #################################################################
    # in the first row we plot the IQ plane with the quadrature data
    # and the principal axes.
    fig.add_traces(
        plot_iq_pca(data, qubit),
        rows=1,
        cols=1,
    )

    #################################################################
    # in the second row we plot the signal projection along the principal axis
    # we computed the fit on.
    principal_signal = pca_signal[:, 0]
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=principal_signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
            mode="markers",
        ),
        row=2,
        col=1,
    )
    #################################################################
    # in the third row we plot the signal projection along the remaining axis.
    residual_signal = pca_signal[:, 1]
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=residual_signal,
            opacity=1,
            name="Residual Signal",
            showlegend=True,
            legendgroup="Residual Signal",
            mode="markers",
        ),
        row=3,
        col=1,
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            500,
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                mode="lines",
                marker_color="rgb(255, 130, 67)",
            ),
            row=2,
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
            xaxis_title="I [a.u.]",
            yaxis_title="Q [a.u.]",
            yaxis2_title="Principal Axis Signal [a.u.]",
            xaxis2_title=title,
            yaxis3_title="Residual Signal [a.u.]",
            xaxis3_title=title,
        )

    fig.update_layout(
        height=800,
    )

    return [fig], fitting_report


def plot_probabilities(data, qubit, fit, rx90):
    quantity, title, fitting = extract_rabi(data)
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
                mode="markers",
            ),
            go.Scatter(
                x=np.concatenate((rabi_parameters, rabi_parameters[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line={"color": COLORBAND_LINE},
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            500,
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *params),
                name="Fit",
                mode="lines",
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

    return [fig], fitting_report


def extract_rabi(data):
    """
    Extract Rabi fit info.
    """
    if "RabiAmplitude" in data.__class__.__name__:
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
) -> tuple[PulseSequence, dict, dict, dict, dict]:
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
    x,
    y,
    guess,
    sigma=None,
) -> tuple[list[float], list[float], float]:
    popt, perr = curve_fit(
        rabi_length_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            [-np.inf, -np.inf, 0, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
        sigma=sigma,
    )

    popt = np.asarray(popt).tolist()
    perr = np.sqrt(np.diag(perr)).tolist()

    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])
    return popt, perr, pi_pulse_parameter


def fit_amplitude_function(
    x,
    y,
    guess,
    sigma=None,
) -> tuple[list[float], list[float], float]:
    popt, perr = curve_fit(
        rabi_amplitude_function,
        x,
        y,
        p0=guess,
        maxfev=100000,
        bounds=(
            [-np.inf, -np.inf, 0, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        sigma=sigma,
    )

    popt = np.asarray(popt).tolist()
    perr = np.sqrt(np.diag(perr)).tolist()

    pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])

    return popt, perr, pi_pulse_parameter
