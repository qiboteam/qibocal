# -*- coding: utf-8 -*-
import os.path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Data, Dataset
from qcvv.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# Flipping
def flips_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"flips": "dimensionless"})

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
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="T1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="T1",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("flips", "dimensionless")),
            max(data.get_values("flips", "dimensionless")),
            20,
        )
        for j in range(2):
            params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
            fig.add_trace(
                go.Scatter(
                    x=timerange,
                    y=flipping(
                        timerange,
                        data_fit.df["popt0"][j],
                        data_fit.df["popt1"][j],
                        data_fit.df["popt2"][j],
                        data_fit.df["popt3"][j],
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
                    x=0 + j / 2,
                    y=-0.25,
                    showarrow=False,
                    text=f"Estimated {params[0]} is {data_fit.df[params[0]][j]:.4f}",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=0 + j / 2,
                    y=-0.30,
                    showarrow=False,
                    text=f"Estimated {params[1]} is {data_fit.df[params[1]][j]:.3f}",
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
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Flips (dimensionless)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# allXY
def prob_gate_iteration(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            quantities={
                "probability": "dimensionless",
                "gateNumber": "dimensionless",
                "beta_param": "dimensionless",
            }
        )

    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY_qubit{qubit}",),
    )

    gates = len(data.get_values("gateNumber", "dimensionless"))
    # print(gates)
    import numpy as np

    for n in range(gates // 21):
        data_start = n * 21
        data_end = data_start + 21
        beta_param = np.array(data.get_values("beta_param", "dimensionless"))[
            data_start
        ]
        gates = np.array(data.get_values("gateNumber", "dimensionless"))[
            data_start:data_end
        ]
        probabilities = np.array(data.get_values("probability", "dimensionless"))[
            data_start:data_end
        ]
        c = "#" + "{:06x}".format(n * 823000)
        fig.add_trace(
            go.Scatter(
                x=gates,
                y=probabilities,
                mode="markers+lines",
                line=dict(color=c),
                name=f"beta_parameter = {beta_param}",
                marker_size=16,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequene number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# allXY
def prob_gate(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            quantities={"probability": "dimensionless", "gateNumber": "dimensionless"}
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY_qubit{qubit}",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gateNumber", "dimensionless"),
            y=data.get_values("probability", "dimensionless"),
            mode="markers",
            name="Probabilities",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequene number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig
