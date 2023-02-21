import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import rabi
from qibocal.plots.utils import get_color, get_data_subfolders, grouped_by_mean


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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(quantities={"time": "ns"}, options=["qubit", "iteration"])

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
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
        times = data.df["time"].pint.to("ns").pint.magnitude.unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["time"].pint.to("ns").pint.magnitude,
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
                    x=iteration_data["time"].pint.to("ns").pint.magnitude,
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
            data.df = data.df.drop(columns=["iteration"])
            unique_times, mean_measurements = grouped_by_mean(data.df, "time", "MSR")

            fig.add_trace(
                go.Scatter(
                    x=unique_times,
                    y=mean_measurements * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )
            unique_times, mean_phases = grouped_by_mean(data.df, "time", "phase")
            fig.add_trace(
                go.Scatter(
                    x=unique_times,
                    y=mean_phases,  # pylint: disable=E1101
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
                min(data.get_values("time", "ns")),
                max(data.get_values("time", "ns")),
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
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                        data_fit.get_values("popt4"),
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"gain", "dimensionless"}, options=["qubit", "iteration"]
            )

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
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
        gains = data.df["gain"].pint.to("dimensionless").pint.magnitude.unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["gain"].pint.to("dimensionless").pint.magnitude,
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
                    x=iteration_data["gain"].pint.to("dimensionless").pint.magnitude,
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
            data.df = data.df.drop(columns=["iteration"])
            unique_gains, mean_measurements = grouped_by_mean(data.df, "gain", "MSR")

            fig.add_trace(
                go.Scatter(
                    x=unique_gains,
                    y=mean_measurements * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

            unique_gains, mean_phases = grouped_by_mean(data.df, "gain", "phase")

            fig.add_trace(
                go.Scatter(
                    x=unique_gains,
                    y=mean_phases,
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
                min(data.get_values("gain", "dimensionless")),
                max(data.get_values("gain", "dimensionless")),
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
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                        data_fit.get_values("popt4"),
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"amplitude", "dimensionless"},
                options=["qubit", "iteration"],
            )
        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
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
        amplitudes = (
            data.df["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
        )
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["amplitude"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
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
                    x=iteration_data["amplitude"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
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
            data.df = data.df.drop(columns=["iteration"])
            unique_amplitudes, mean_measurements = grouped_by_mean(
                data.df, "amplitude", "MSR"
            )
            fig.add_trace(
                go.Scatter(
                    x=unique_amplitudes,
                    y=mean_measurements * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )
            unique_amplitudes, mean_phases = grouped_by_mean(
                data.df, "amplitude", "phase"
            )
            fig.add_trace(
                go.Scatter(
                    x=unique_amplitudes,
                    y=mean_phases,
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
                min(data.get_values("amplitude", "dimensionless")),
                max(data.get_values("amplitude", "dimensionless")),
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
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                        data_fit.get_values("popt4"),
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"duration": "ns", "gain": "dimensionless"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        durations = data.df["duration"].pint.to("ns").pint.magnitude.unique()
        gains = data.df["gain"].pint.to("dimensionless").pint.magnitude.unique()

        averaged_data = data.df.drop(columns=["i", "q", "qubit", "iteration"])

        if len(iterations) > 1:
            (
                unique_durations,
                unique_gains,
                mean_measurements,
                mean_phases,
            ) = grouped_by_mean(averaged_data, "duration", "MSR", "gain", "phase")
        else:
            unique_durations = averaged_data["duration"].pint.to("ns").pint.magnitude
            unique_gains = averaged_data["gain"].pint.to("dimensionless").pint.magnitude
            mean_measurements = averaged_data["MSR"].pint.to("V").pint.magnitude
            mean_phases = averaged_data["phase"].pint.to("rad").pint.magnitude

        fig.add_trace(
            go.Heatmap(
                x=unique_durations,
                y=unique_gains,
                z=mean_measurements,
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
                x=unique_durations,
                y=unique_gains,
                z=mean_phases,
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"duration": "ns", "amplitude": "dimensionless"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        durations = data.df["duration"].pint.to("ns").pint.magnitude.unique()
        amplitudes = (
            data.df["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
        )

        averaged_data = data.df.drop(columns=["i", "q", "qubit", "iteration"])

        if len(iterations) > 1:
            (
                unique_durations,
                unique_amplitudes,
                mean_measurements,
                mean_phases,
            ) = grouped_by_mean(averaged_data, "duration", "MSR", "amplitude", "phase")
        else:
            unique_durations = averaged_data["duration"].pint.to("ns").pint.magnitude
            unique_amplitudes = (
                averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude
            )
            mean_measurements = averaged_data["MSR"].pint.to("V").pint.magnitude
            mean_phases = averaged_data["phase"].pint.to("rad").pint.magnitude

        fig.add_trace(
            go.Heatmap(
                x=unique_durations,
                y=unique_amplitudes,
                z=mean_measurements,
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
                x=unique_durations,
                y=unique_amplitudes,
                z=mean_phases,
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
