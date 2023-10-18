import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import V_TO_UV, table_dict, table_html


def rabi_amplitude_fit(x, p0, p1, p2, p3):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3)


def rabi_length_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def plot(data, qubit, fit):
    if data.__class__.__name__ == "RabiAmplitudeData":
        quantity = "amp"
        title = "Amplitude (dimensionless)"
        fitting = rabi_amplitude_fit
    elif data.__class__.__name__ == "RabiLengthData":
        quantity = "length"
        title = "Time (ns)"
        fitting = rabi_length_fit

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    qubit_data = data[qubit]

    rabi_parameters = getattr(qubit_data, quantity)
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=qubit_data.msr * V_TO_UV,
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
                y=fitting(rabi_parameter_range, *params) * V_TO_UV,
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
            yaxis_title="MSR (uV)",
            xaxis2_title=title,
            yaxis2_title="Phase (rad)",
        )

    figures.append(fig)

    return figures, fitting_report


def plot_proba(data, qubit, fit):
    if data.__class__.__name__ == "RabiAmplitudeData":
        quantity = "amp"
        title = "Amplitude (dimensionless)"
        fitting = rabi_amplitude_fit
    elif data.__class__.__name__ == "RabiLengthData":
        quantity = "length"
        title = "Time (ns)"
        fitting = rabi_length_fit

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("Probability",),
    )

    qubit_data = data[qubit]

    rabi_parameters = getattr(qubit_data, quantity)
    fig.add_trace(
        go.Scatter(
            x=rabi_parameters,
            y=qubit_data.prob,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
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
            yaxis_title="Excited state probability",
        )

    figures.append(fig)

    return figures, fitting_report
