# -*- coding: utf-8 -*-
import os.path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Data, Dataset
from qcvv.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
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
            y=data.get_values("phase", "deg"),
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
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        # add annotation for label[0] -> pi_pulse_duration
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        # add annotation for label[0] -> rabi_oscillations_pi_pulse_max_voltage
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        # add annotation for label[0] -> rabi_oscillations_pi_pulse_max_voltage
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[2]} is {data_fit.df[params[2]][0]:.1f} ns.",
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
        yaxis2_title="Phase (deg)",
    )
    return fig


# T1
def t1_time_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("MSR", "uV"),
            name="T1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "deg"),
            name="T1",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=exp(
                    timerange,
                    data_fit.df["popt0"][0],
                    data_fit.df["popt1"][0],
                    data_fit.df["popt2"][0],
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
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns.",
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
        yaxis2_title="Phase (deg)",
    )
    return fig


# For Ramsey oscillations
def time_msr(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("wait", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Ramsey MSR",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("wait", "ns"),
            y=data.get_values("phase", "deg"),
            name="Ramsey Phase",
        ),
        row=1,
        col=2,
    )

    for j in range(2):
        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            timerange = np.linspace(
                min(data.get_values("wait", "ns")),
                max(data.get_values("wait", "ns")),
                len(data.get_values("wait", "ns")) * 2,
            )
            params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
            fig.add_trace(
                go.Scatter(
                    x=timerange,
                    y=ramsey(
                        timerange,
                        data_fit.df["popt0"][j],
                        data_fit.df["popt1"][j],
                        data_fit.df["popt2"][j],
                        data_fit.df["popt3"][j],
                        data_fit.df["popt4"][j],
                    ),
                    name="Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=j + 1,
            )

            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=j / 2,
                    y=-0.30,
                    showarrow=False,
                    text=f"Estimated {params[1]} is {data_fit.df[params[1]][j]:.3f} Hz.",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )

            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=j / 2,
                    y=-0.20,
                    showarrow=False,
                    text=f"Estimated {params[0]} is {data_fit.df[params[0]][j]:.1f} ns",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )

            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=j / 2,
                    y=-0.25,
                    showarrow=False,
                    text=f"Estimated {params[2]} is {data_fit.df[params[2]][j]:.3f} Hz",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (deg)",
    )
    return fig
