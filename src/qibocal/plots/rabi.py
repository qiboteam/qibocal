import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import rabi
from qibocal.plots.utils import get_data_subfolders


# Rabi oscillations pulse length
def time_msr_phase(folder, routine, qubit, format):

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
    i = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(quantities={"Time": "ns"})

        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
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
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=data.get_values("Time", "ns"),
                y=data.get_values("MSR", "uV"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("Time", "ns"),
                y=data.get_values("phase", "rad"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=2,
        )
        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            timerange = np.linspace(
                min(data.get_values("Time", "ns")),
                max(data.get_values("Time", "ns")),
                2 * len(data),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f} ns<br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.1f} uV.<br><br>"
            )

        i += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Rabi oscillations pulse gain
def gain_msr_phase(folder, routine, qubit, format):

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
    i = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(quantities={"gain", "dimensionless"})

        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
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
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=data.get_values("gain", "dimensionless"),
                y=data.get_values("MSR", "uV"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("gain", "dimensionless"),
                y=data.get_values("phase", "rad"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=2,
        )

        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            gainrange = np.linspace(
                min(data.get_values("gain", "dimensionless")),
                max(data.get_values("gain", "dimensionless")),
                2 * len(data),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f}<br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.4f} uV<br><br>"
            )

        i += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


# Rabi oscillations pulse amplitude
def amplitude_msr_phase(folder, routine, qubit, format):

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
    i = 0
    fitting_report = ""
    for subfolder in subfolders:

        try:
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(quantities={"amplitude", "dimensionless"})
        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
        except:
            data_fit = DataUnits()

        fig.add_trace(
            go.Scatter(
                x=data.get_values("amplitude", "dimensionless"),
                y=data.get_values("MSR", "uV"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("amplitude", "dimensionless"),
                y=data.get_values("phase", "rad"),
                name=f"Rabi q{qubit}/r{i}",
            ),
            row=1,
            col=2,
        )

        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            amplituderange = np.linspace(
                min(data.get_values("amplitude", "dimensionless")),
                max(data.get_values("amplitude", "dimensionless")),
                2 * len(data),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.3f} uV.<br>q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.4f}<br><br>"
            )

        i += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


# Rabi pulse length and gain
def duration_gain_msr_phase(folder, routine, qubit, format):

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
    i = 1
    for subfolder in subfolders:
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("gain", "dimensionless"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("gain", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"]["yaxis"]["title"] = "gain (dimensionless)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"

        i += 1

    return fig


# Rabi pulse length and amplitude
def duration_amplitude_msr_phase(folder, routine, qubit, format):

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

    i = 1
    for subfolder in subfolders:
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("amplitude", "dimensionless"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("amplitude", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"]["yaxis"]["title"] = "A (dimensionless)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"

        i += 1

    return fig
