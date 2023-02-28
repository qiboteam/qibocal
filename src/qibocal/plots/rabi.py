import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import rabi
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# Rabi oscillations pulse length
def time_msr_phase(folder, routine, qubit, format):
    figures = []

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
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(quantities={"time": "ns"}, options=["qubit", "iteration"])

        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "label1",
                    "label2",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        times = data.df["time"].unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["time"],
                    y=iteration_data["MSR"] * 1e6,
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
                    x=iteration_data["time"],
                    y=iteration_data["phase"],
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
                    x=times,
                    y=data.df.groupby("time")["MSR"].mean()
                    * 1e6,  # pylint: disable=E1101
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
                    x=times,
                    y=data.df.groupby("time")["phase"].mean(),  # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=2,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            timerange = np.linspace(
                min(data.df["time"]),
                max(data.df["time"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=timerange,
                    y=rabi(
                        timerange,
                        data_fit.df["popt0"][0],
                        data_fit.df["popt1"][0],
                        data_fit.df["popt2"][0],
                        data_fit.df["popt3"][0],
                        data_fit.df["popt4"][0],
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_duration: {params['pi_pulse_duration']:.2f} ns<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_peak_voltage: {params['pi_pulse_peak_voltage']:,.0f} uV.<br><br>"
                )
            )

        report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


# Rabi oscillations pulse gain
def gain_msr_phase(folder, routine, qubit, format):
    figures = []

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
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"gain", "dimensionless"}, options=["qubit", "iteration"]
            )

        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "label1",
                    "label2",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        gains = data.df["gain"].unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["gain"],
                    y=iteration_data["MSR"] * 1e6,
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
                    x=iteration_data["gain"],
                    y=iteration_data["phase"],
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
                    x=gains,
                    y=data.df.groupby("gain")["MSR"].mean()
                    * 1e6,  # pylint: disable=E1101
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
                    x=gains,
                    y=data.df.groupby("gain")["phase"].mean(),  # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=2,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            gainrange = np.linspace(
                min(data.df["gain"]),
                max(data.df["gain"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=gainrange,
                    y=rabi(
                        gainrange,
                        data_fit.df["popt0"][0],
                        data_fit.df["popt1"][0],
                        data_fit.df["popt2"][0],
                        data_fit.df["popt3"][0],
                        data_fit.df["popt4"][0],
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_gain: {params['pi_pulse_gain']:.3f}<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_peak_voltage: {params['pi_pulse_peak_voltage']:,.0f} uV.<br><br>"
                )
            )

        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Gain (dimensionless)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


# Rabi oscillations pulse amplitude
def amplitude_msr_phase(folder, routine, qubit, format):
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
    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"amplitude", "dimensionless"},
                options=["qubit", "iteration"],
            )
        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "label1",
                    "label2",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        amplitudes = data.df["amplitude"].unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["amplitude"],
                    y=iteration_data["MSR"] * 1e6,
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
                    x=iteration_data["amplitude"],
                    y=iteration_data["phase"],
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
                    x=amplitudes,
                    y=data.df.groupby("amplitude")["MSR"].mean()
                    * 1e6,  # pylint: disable=E1101
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
                    x=amplitudes,
                    y=data.df.groupby("amplitude")[  # pylint: disable=E1101
                        "phase"
                    ].mean(),
                    marker_color=get_color(report_n),
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=2,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            amplituderange = np.linspace(
                min(data.df["amplitude"]),
                max(data.df["amplitude"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=amplituderange,
                    y=rabi(
                        amplituderange,
                        data_fit.df["popt0"][0],
                        data_fit.df["popt1"][0],
                        data_fit.df["popt2"][0],
                        data_fit.df["popt3"][0],
                        data_fit.df["popt4"][0],
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color="rgb(255, 130, 67)",
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_amplitude: {params['pi_pulse_amplitude']:.3f}<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | pi_pulse_peak_voltage: {params['pi_pulse_peak_voltage']:,.0f} uV.<br><br>"
                )
            )

        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Amplitude (dimensionless)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


# Rabi pulse length and gain
def duration_gain_msr_phase(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"duration": "ns", "gain": "dimensionless"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        durations = data.df["duration"].unique()
        gains = data.df["gain"].unique()

        averaged_data = data.df.drop(columns=["i", "q", "qubit", "iteration"])
        averaged_data = data.df.groupby(
            ["duration", "gain"], as_index=False
        ).mean()  # pylint: disable=E1101

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["duration"],
                y=averaged_data["gain"],
                z=averaged_data["MSR"] * 1e6,
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Duration (ns)", row=1 + report_n, col=1
        )
        fig.update_yaxes(title_text="Gain (dimensionless)", row=1 + report_n, col=1)
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["duration"],
                y=averaged_data["gain"],
                z=averaged_data["phase"],
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Duration (ns)", row=1 + report_n, col=2
        )
        fig.update_yaxes(title_text="Gain (dimensionless)", row=1 + report_n, col=2)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report


# Rabi pulse length and amplitude
def duration_amplitude_msr_phase(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"duration": "ns", "amplitude": "dimensionless"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        durations = data.df["duration"].unique()
        amplitudes = data.df["amplitude"].unique()

        averaged_data = data.df.drop(columns=["i", "q", "qubit", "iteration"])

        averaged_data = data.df.groupby(  # pylint: disable=E1101
            ["duration", "amplitude"], as_index=False
        ).mean()

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["duration"],
                y=averaged_data["amplitude"],
                z=averaged_data["MSR"] * 1e6,
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Duration (ns)", row=1 + report_n, col=1
        )
        fig.update_yaxes(
            title_text="Amplitude (dimensionless)", row=1 + report_n, col=1
        )
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["duration"],
                y=averaged_data["amplitude"],
                z=averaged_data["phase"],
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Duration (ns)", row=1 + report_n, col=2
        )
        fig.update_yaxes(
            title_text="Amplitude (dimensionless)", row=1 + report_n, col=2
        )
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report
