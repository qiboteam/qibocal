import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.protocols.utils import (
    COLORBAND,
    COLORBAND_LINE,
    table_dict,
    table_html,
)

from .fitting import extract_rabi


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
            mode="markers",
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
            mode="markers",
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
                mode="lines",
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
            2 * len(rabi_parameters),
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

    figures.append(fig)

    return figures, fitting_report
