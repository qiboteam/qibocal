import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import rabi


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = DataUnits(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
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

    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("time"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["time"].pint.to("ns").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                opacity=0.3,
                name="MSR",
                showlegend=not bool(i),
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
                name="phase",
                opacity=0.5,
                showlegend=not bool(i),
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
            name="average MSR",
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
            name="average phase",
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
                name="Fitted MSR",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
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


def gain_msr_phase(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = DataUnits(quantities={"gain", "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
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

    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("gain"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["gain"].pint.to("dimensionless").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                opacity=0.3,
                name="MSR",
                showlegend=not bool(i),
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
                name="phase",
                opacity=0.5,
                showlegend=not bool(i),
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
            name="average MSR",
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
            name="average phase",
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
                name="Fitted MSR",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f} uV",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def amplitude_msr_phase(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = DataUnits(quantities={"amplitude", "dimensionless"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("amplitude"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["amplitude"].pint.to("dimensionless").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                opacity=0.3,
                name="MSR",
                showlegend=not bool(i),
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
                name="phase",
                opacity=0.5,
                showlegend=not bool(i),
                legendgroup="group2",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=data.df.amplitude.drop_duplicates()  # pylint: disable=E1101
            .pint.to("dimensionless")
            .pint.magnitude,
            y=data.df.groupby("amplitude")["MSR"]  # pylint: disable=E1101
            .mean()
            .pint.to("uV")
            .pint.magnitude,
            name="average MSR",
            marker_color="rgb(100, 0, 255)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.df.amplitude.drop_duplicates()  # pylint: disable=E1101
            .pint.to("dimensionless")
            .pint.magnitude,
            y=data.df.groupby("amplitude")["phase"]  # pylint: disable=E1101
            .mean()
            .pint.to("rad")
            .pint.magnitude,
            name="average phase",
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
                name="Fitted MSR",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def duration_gain_msr_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, "data")
    data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
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

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("gain", "dimensionless"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("gain", "dimensionless"),
            z=data.get_values("phase", "rad"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="gain (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="gain (dimensionless)",
    )
    return fig


def duration_amplitude_msr_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, "data")
    data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
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

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("amplitude", "dimensionless"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("amplitude", "dimensionless"),
            z=data.get_values("phase", "rad"),
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
    return fig
