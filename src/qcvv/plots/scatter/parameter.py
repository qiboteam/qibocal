# -*- coding: utf-8 -*-
import os.path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Data, Dataset
from qcvv.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# beta param tuning
def msr_beta(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=(f"beta_param_tuning_qubit{qubit}",),
    )

    c = "#6597aa"
    fig.add_trace(
        go.Scatter(
            x=data.get_values("beta_param", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            line=dict(color=c),
            mode="markers",
            name="[Rx(pi/2) - Ry(pi)] - [Ry(pi) - Rx(pi/2)]",
        ),
        row=1,
        col=1,
    )
    # add fitting traces
    if len(data) > 0 and len(data_fit) > 0:
        beta_param = np.linspace(
            min(data.get_values("beta_param", "dimensionless")),
            max(data.get_values("beta_param", "dimensionless")),
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
        fig.add_trace(
            go.Scatter(
                x=beta_param,
                y=cos(
                    beta_param,
                    data_fit.df["popt0"][0],
                    data_fit.df["popt1"][0],
                    data_fit.df["popt2"][0],
                    data_fit.df["popt3"][0],
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
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV]",
    )
    return fig


# For calibrate qubit states
def exc_gnd(folder, routine, qubit, format):

    import os.path

    file_exc = f"{folder}/data/{routine}/data_exc_q{qubit}.csv"
    if os.path.exists(file_exc):
        data_exc = Dataset.load_data(folder, routine, format, f"data_exc_q{qubit}")

        fig = make_subplots(
            rows=1,
            cols=1,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=("Calibrate qubit states",),
        )

        fig.add_trace(
            go.Scatter(
                x=data_exc.get_values("i", "V"),
                y=data_exc.get_values("q", "V"),
                name="exc_state",
                mode="markers",
                marker=dict(size=3, color="lightcoral"),
            ),
            row=1,
            col=1,
        )

    file_gnd = f"{folder}/data/{routine}/data_gnd_q{qubit}.csv"
    if os.path.exists(file_gnd):
        data_gnd = Dataset.load_data(folder, routine, format, f"data_gnd_q{qubit}")

        fig.add_trace(
            go.Scatter(
                x=data_gnd.get_values("i", "V"),
                y=data_gnd.get_values("q", "V"),
                name="gnd state",
                mode="markers",
                marker=dict(size=3, color="skyblue"),
            ),
            row=1,
            col=1,
        )

    file_exc = f"{folder}/data/{routine}/data_exc_q{qubit}.csv"
    if os.path.exists(file_exc):
        i_exc = data_exc.get_values("i", "V")
        q_exc = data_exc.get_values("q", "V")

        i_mean_exc = i_exc.mean()
        q_mean_exc = q_exc.mean()
        iq_mean_exc = complex(i_mean_exc, q_mean_exc)
        mod_iq_exc = abs(iq_mean_exc) * 1e6

        fig.add_trace(
            go.Scatter(
                x=[i_mean_exc],
                y=[q_mean_exc],
                name=f" state1_voltage: {mod_iq_exc} <br> mean_exc_state: {iq_mean_exc}",
                mode="markers",
                marker=dict(size=10, color="red"),
            ),
            row=1,
            col=1,
        )

    file_gnd = f"{folder}/data/{routine}/data_gnd_q{qubit}.csv"
    if os.path.exists(file_gnd):
        i_gnd = data_gnd.get_values("i", "V")
        q_gnd = data_gnd.get_values("q", "V")

        i_mean_gnd = i_gnd.mean()
        q_mean_gnd = q_gnd.mean()
        iq_mean_gnd = complex(i_mean_gnd, q_mean_gnd)
        mod_iq_gnd = abs(iq_mean_gnd) * 1e6

        fig.add_trace(
            go.Scatter(
                x=[i_mean_gnd],
                y=[q_mean_gnd],
                name=f" state0_voltage: {mod_iq_gnd} <br> mean_gnd_state: {iq_mean_gnd}",
                mode="markers",
                marker=dict(size=10, color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="i (V)",
            yaxis_title="q (V)",
            width=1000,
        )

    return fig
