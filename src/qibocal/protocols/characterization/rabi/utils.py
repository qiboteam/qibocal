import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html


def rabi_amplitude_function(x, p0, p1, p2, p3):
    """
    Fit function of rabi amplitude signal experiment.

    Args:
        x: Input data.
        p0: Offset.
        p1: Oscillation Amplitude.
        p2: Period.
        p3: Phase.
    """
    return p0 + p1 * np.sin(2 * np.pi * x / p2 + p3)


def rabi_length_function(x, p0, p1, p2, p3, p4):
    """
    Fit function of rabi length signal experiment.

    Args:
        x: Input data.
        p0: Offset.
        p1: Oscillation Amplitude.
        p2: Period.
        p3: Phase.
        p4: T2.
    """
    return p0 + p1 * np.cos(2 * np.pi * x / p2 + p3) * np.exp(-x * p4)


def plot(data, qubit, fit):
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
            "phase (rad)",
        ),
    )

    qubit_data = data[qubit]

    rabi_parameters = getattr(qubit_data, quantity)
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=qubit_data.signal,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
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

        fitting_report = table_html(
            table_dict(
                qubit,
                ["Pi pulse amplitude", "Pi pulse length"],
                [np.round(fit.amplitude[qubit], 3), np.round(fit.length[qubit], 3)],
            )
        )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title=title,
            yaxis_title="Signal [a.u.]",
            xaxis2_title=title,
            yaxis2_title="Phase (rad)",
        )

    figures.append(fig)

    return figures, fitting_report


def plot_probabilities(data, qubit, fit):
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

        fitting_report = table_html(
            table_dict(
                qubit,
                ["Pi pulse amplitude", "Pi pulse length", "chi2 reduced"],
                [fit.amplitude[qubit], fit.length[qubit], fit.chi2[qubit]],
                display_error=True,
            )
        )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title=title,
            yaxis_title="Excited state probability",
        )

    figures.append(fig)

    return figures, fitting_report


def extract_rabi(data):
    """
    Extract Rabi fit info.
    """
    if data.__class__.__name__ == "RabiAmplitudeData":
        return "amp", "Amplitude (dimensionless)", rabi_amplitude_function
    elif data.__class__.__name__ == "RabiLengthData":
        return "length", "Time (ns)", rabi_length_function
