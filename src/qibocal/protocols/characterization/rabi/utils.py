import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot(data, fit, qubit):
    if data.__class__.__name__ == "RabiAmplitudeData":
        quantity = "amplitude"
        unit = "dimensionless"
        title = "Amplitude (dimensionless)"
        fitting = rabi_amplitude_fit
    elif data.__class__.__name__ == "RabiLengthData":
        quantity = "length"
        unit = "ns"
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

    qubit_data = data.df[data.df["qubit"] == qubit]

    rabi_parameters = qubit_data[quantity].pint.to(unit).pint.magnitude.unique()
    fig.add_trace(
        go.Scatter(
            x=qubit_data[quantity].pint.to(unit).pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
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
            x=qubit_data[quantity].pint.to(unit).pint.magnitude,
            y=qubit_data["phase"].pint.to("rad").pint.magnitude,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            2 * len(data),
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

        fitting_report += (
            f"{qubit} | pi_pulse_amplitude: {float(fit.amplitude[qubit]):.3f}<br>"
        )
        fitting_report += (
            f"{qubit} | pi_pulse_length: {float(fit.length[qubit]):.3f}<br>"
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
