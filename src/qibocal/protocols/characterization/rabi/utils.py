import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color


def rabi(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def fitting(data: DataUnits, label: str) -> list:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()

    rabi_parameters = {}
    fitted_parameters = {}

    if label == "amplitude":
        quantity = "amplitude"
        unit = "dimensionless"
    elif label == "length":
        quantity = "time"
        unit = "ns"

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration", "resonator_type"])
            .groupby(quantity, as_index=False)
            .mean()
        )

        rabi_parameter = qubit_data[quantity].pint.to(unit).pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        if resonator_type == "3D":
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / rabi_parameter.values[np.argmin(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        else:
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / rabi_parameter.values[np.argmax(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        try:
            popt, pcov = curve_fit(
                rabi, rabi_parameter.values, voltages.values, p0=pguess, maxfev=10000
            )
            pi_pulse_parameter = np.abs((1.0 / popt[2]) / 2)
            # pi_pulse_peak_voltage = smooth_dataset.max()
            # t2 = 1.0 / popt[4]  # double check T1

        except:
            log.warning("rabi_fit: the fitting was not succesful")

        rabi_parameters[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = popt

    return rabi_parameters, fitted_parameters


def plot(data, fit, qubit, label):
    if label == "amplitude":
        quantity = "amplitude"
        unit = "dimensionless"
        title = "Amplitude (dimensionless)"
    elif label == "length":
        quantity = "time"
        unit = "ns"
        title = "Time (ns)"

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

    # iterate over multiple data folders
    report_n = 0

    data.df = data.df[data.df["qubit"] == qubit]
    iterations = data.df["iteration"].unique()
    data.df = data.df.drop(columns=["i", "q", "qubit", "resonator_type"])

    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        rabi_parameters = data.df[quantity].pint.to(unit).pint.magnitude.unique()
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data[quantity].pint.to(unit).pint.magnitude,
                y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=iteration_data[quantity].pint.to(unit).pint.magnitude,
                y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=2,
        )
    if len(iterations) > 1:
        data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
        fig.add_trace(
            go.Scatter(
                x=rabi_parameters,
                y=data.df.groupby(quantity)["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                marker_color=get_color(report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=rabi_parameters,
                y=data.df.groupby(quantity)["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                marker_color=get_color(report_n),
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Average",
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
                y=rabi(rabi_parameter_range, *params),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        if label == "amplitude":
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} | pi_pulse_amplitude: {fit.amplitude[qubit]:.3f}<br>"
            )
        elif label == "length":
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} | pi_pulse_length: {fit.length[qubit]:.3f}<br>"
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
