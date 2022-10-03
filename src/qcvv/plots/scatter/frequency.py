# -*- coding: utf-8 -*-
import os.path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Data, Dataset
from qcvv.fitting.utils import (
    cos,
    exp,
    flipping,
    lorenzian,
    lorenzian_diff,
    rabi,
    ramsey,
)


def frequency_msr_phase__all(folder, routine, qubits, format):
    # Method that can plot 1 or multiple qubits
    if not isinstance(qubits, list):
        qubits = [qubits]
        showleg = False
    else:
        showleg = True

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
    fig.update_layout(
        showlegend=showleg,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (V)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )

    # Raw data
    try:
        data = Dataset.load_data(folder, routine, format, f"data")
        d = {
            "x": data.get_values("frequency", "Hz").to_numpy(),
            "MSR": data.get_values("MSR", "V").to_numpy(),
            "Phase": data.get_values("phase", "rad").to_numpy(),
            "q": data.get_values("qubit", "unit").to_numpy(),
        }

    except:
        qubits = []

    for qubit in qubits:
        for i, key in enumerate(["MSR", "Phase"]):
            if key == "Phase":
                d[key][d["q"] == qubit] = np.unwrap(d[key][d["q"] == qubit])
            fig.add_trace(
                go.Scatter(
                    x=d["x"][d["q"] == qubit],
                    y=d[key][d["q"] == qubit],
                    name=f"{key} qubit {qubit}",
                    mode="lines",
                ),
                row=1,
                col=i + 1,
            )

    # Fitting
    try:
        data_fit_msr = Data.load_data(folder, routine, format, f"fit_msr")
        dfit = {
            "fit_amplitude": data_fit_msr.df["fit_amplitude"].to_numpy(),
            "center frequency": data_fit_msr.df["fit_center"].to_numpy(),
            "fit_sigma": data_fit_msr.df["fit_sigma"].to_numpy(),
            "fit_offset": data_fit_msr.df["fit_offset"].to_numpy(),
            "peak value": data_fit_msr.df["peak_value"].to_numpy(),
            "q": data_fit_msr.df["qubit"].to_numpy(),
        }
    except:
        qubits = []

    for qubit in qubits:
        x = np.linspace(
            min(d["x"][d["q"] == qubit]),
            max(d["x"][d["q"] == qubit]),
            2 * len(d["x"][d["q"] == qubit]),
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lorenzian(
                    x,
                    float(dfit["fit_amplitude"][dfit["q"] == qubit]),
                    float(dfit["center frequency"][dfit["q"] == qubit]),
                    float(dfit["fit_sigma"][dfit["q"] == qubit]),
                    float(dfit["fit_offset"][dfit["q"] == qubit]),
                ),
                line=go.scatter.Line(dash="dot"),
                name=f"Fit MSR qubit {qubit}",
            ),
            row=1,
            col=1,
        )
        for i, key in enumerate(["center frequency", "peak value"]):
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=0,
                    y=-0.25 - 0.05 * i,
                    showarrow=False,
                    text=f"The estimated {key} is {dfit[key][dfit['q'] == qubit]} {['Hz','V]'][i]}.",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )

    return fig


def frequency_msr_phase__fast_precision(folder, routine, qubit, format):
    try:
        data_fast = Dataset.load_data(folder, routine, format, f"fast_sweep")
    except:
        data_fast = Dataset(quantities={"frequency": "Hz"})
    try:
        data_precision = Dataset.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
    except:
        data_precision = Dataset(quantities={"frequency": "Hz"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "fit_amplitude",
                "fit_center",
                "fit_sigma",
                "fit_offset",
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
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("MSR", "uV"),
            name="Fast",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("phase", "rad"),
            name="Fast",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("MSR", "uV"),
            name="Precision",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("phase", "rad"),
            name="Precision",
        ),
        row=1,
        col=2,
    )
    if len(data_fast) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_fast.get_values("frequency", "GHz")),
            max(data_fast.get_values("frequency", "GHz")),
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.df["fit_amplitude"][0],
                    data_fit.df["fit_center"][0],
                    data_fit.df["fit_sigma"][0],
                    data_fit.df["fit_offset"][0],
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
                y=-0.25,
                showarrow=False,
                text=f"The estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} Hz.",
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
                text=f"The estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def frequency_attenuation_msr_phase__cut(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

    fig = go.Figure()
    # index data on a specific attenuation value
    smalldf = data.df[data.get_values("attenuation", "dB") == plot1d_attenuation].copy()
    # split multiple software averages to different datasets
    datasets = []
    while len(smalldf):
        datasets.append(smalldf.drop_duplicates("frequency"))
        smalldf.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("V").pint.magnitude,
            ),
        )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


def frequency_fidelity(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"frequency": "Hz", "fidelity": "dimensionless"})

    fig = go.Figure(
        go.Scatter(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("fidelity", "dimensionless"),
        )
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Fidelity (prob)",
    )
    return fig
