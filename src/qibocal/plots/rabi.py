import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import rabi


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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

    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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

    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
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
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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
