import os.path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import landscape
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


def landscape_2q_gate(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    try:
        data_fit = load_data(folder, subfolder, routine, format, "fits")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "qubit",
                "setup",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (uV) - Low Frequency",  # TODO: change this to <Z>
            "MSR (uV) - High Frequency",
        ),
    )

    fitting_report = ""
    column = 0
    for qubit in (lowfreq, highfreq):
        filter = (data.df["target_qubit"] == qubit) & (data.df["qubit"] == qubit)
        thetas = data.df[filter]["theta"].unique()
        column += 1
        color = 0
        offset = {}
        for setup in ("I", "X"):
            color += 1
            fig.add_trace(
                go.Scatter(
                    x=data.get_values("theta", "rad")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    y=data.get_values("MSR", "uV")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    name=f"q{qubit} {setup} Data",
                    marker_color=get_color(2 * column + color),
                ),
                row=1,
                col=column,
            )

            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            params = data_fit.df[
                (data_fit.df["qubit"] == qubit) & (data_fit.df["setup"] == setup)
            ].to_dict(orient="records")[0]
            if (params["popt0"], params["popt1"], params["popt2"]) != (0, 0, 0):
                fig.add_trace(
                    go.Scatter(
                        x=angle_range,
                        y=landscape(
                            angle_range,
                            float(params["popt0"]),
                            float(params["popt1"]),
                            float(params["popt2"]),
                        ),
                        name=f"q{qubit} {setup} Fit",
                        line=go.scatter.Line(dash="dot"),
                        marker_color=get_color(2 * column + color),
                    ),
                    row=1,
                    col=column,
                )
                offset[setup] = params["popt2"]
                fitting_report += (
                    f"q{qubit} {setup} | offset: {offset[setup]:,.3f} rad<br>"
                )
        if "X" in offset and "I" in offset:
            fitting_report += (
                f"q{qubit} | Z rotation: {offset['X'] - offset['I']:,.3f} rad<br>"
            )
        fitting_report += "<br>"

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="theta (rad)",
        yaxis_title="MSR (uV)",
        xaxis2_title="theta (rad)",
        yaxis2_title="MSR (uV)",
    )

    return [fig], fitting_report


def duration_amplitude_msr_flux_pulse(folder, routine, qubit, format):
    fitting_report = "No fitting data"
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V) - High Frequency",
            "MSR (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("MSR", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("MSR", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig], fitting_report


def duration_amplitude_I_flux_pulse(folder, routine, qubit, format):
    fitting_report = "No fitting data"
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "I (V) - High Frequency",
            "I (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("i", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("i", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig], fitting_report


def duration_amplitude_Q_flux_pulse(folder, routine, qubit, format):
    fitting_report = "No fitting data"
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Q (V) - High Frequency",
            "Q (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("q", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("q", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig], fitting_report
