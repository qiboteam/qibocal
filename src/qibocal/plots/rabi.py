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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"time": "ns"})

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

        datasets = []
        copy = data.df.copy()
        for j in range(len(copy)):
            datasets.append(copy.drop_duplicates("time"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["time"].pint.to("ns").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"MSR q{qubit}/r{i}",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["time"].pint.to("ns").pint.magnitude,
                    y=datasets[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"phase q{qubit}/r{i}",
                    opacity=0.5,
                    showlegend=not bool(j),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.time.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("time")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"avg MSR q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.df.time.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("time")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"avg phase q{qubit}/r{i}",
                marker_color="rgb(102, 180, 71)",
            ),
            row=1,
            col=2,
        )
        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            timerange = np.linspace(
                min(data.get_values("time", "ns")),
                max(data.get_values("time", "ns")),
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
        xaxis_title="time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="time (ns)",
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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"gain": "dimensionless"})

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

        datasets = []
        copy = data.df.copy()
        for j in range(len(copy)):
            datasets.append(copy.drop_duplicates("gain"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["gain"].pint.to("dimensionless").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"MSR q{qubit}/r{i}",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["gain"].pint.to("dimensionless").pint.magnitude,
                    y=datasets[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"phase q{qubit}/r{i}",
                    opacity=0.5,
                    showlegend=not bool(j),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.gain.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("gain")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"avg MSR q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.df.gain.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("gain")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"avg phase q{qubit}/r{i}",
                marker_color="rgb(102, 180, 71)",
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
                f"q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f} <br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.1f} uV.<br><br>"
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
        xaxis_title="gain (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="gain (dimensionless)",
        yaxis2_title="Phase (rad)",
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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"amplitude": "dimensionless"})

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

        datasets = []
        copy = data.df.copy()
        for j in range(len(copy)):
            datasets.append(copy.drop_duplicates("amplitude"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["amplitude"].pint.to("dimensionless").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"MSR q{qubit}/r{i}",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["amplitude"].pint.to("dimensionless").pint.magnitude,
                    y=datasets[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"phase q{qubit}/r{i}",
                    opacity=0.5,
                    showlegend=not bool(j),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.amplitude.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("amplitude")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"avg MSR q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.df.amplitude.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("amplitude")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"avg phase q{qubit}/r{i}",
                marker_color="rgb(102, 180, 71)",
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
                f"q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f} <br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.1f} uV.<br><br>"
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
        xaxis_title="amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="amplitude (dimensionless)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# # Rabi pulse length and gain
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
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(
                quantities={"duration": "ns", "gain": "dimensionless"},
                options=["qubit"],
            )

        size = len(data.df.duration.drop_duplicates()) * len(
            data.df.gain.drop_duplicates()  # pylint: disable=E1101
        )

        fig.add_trace(
            go.Heatmap(
                x=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .duration.mean()
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .gain.mean()
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .MSR.mean()
                .pint.to("V")
                .pint.magnitude,
                colorbar_x=0.46,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .duration.mean()
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .gain.mean()
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .phase.mean()
                .pint.to("rad")
                .pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1,
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
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(
                quantities={"duration": "ns", "amplitude": "dimensionless"},
                options=["qubit"],
            )

        size = len(data.df.duration.drop_duplicates()) * len(
            data.df.amplitude.drop_duplicates()  # pylint: disable=E1101
        )

        fig.add_trace(
            go.Heatmap(
                x=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .duration.mean()
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .amplitude.mean()
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .MSR.mean()
                .pint.to("V")
                .pint.magnitude,
                colorbar_x=0.46,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .duration.mean()
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .amplitude.mean()
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .phase.mean()
                .pint.to("rad")
                .pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1,
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
