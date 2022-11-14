import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import lorenzian


def frequency_msr_phase__fast_precision(folder, routine, qubit, format):
    try:
        data_fast = DataUnits.load_data(folder, routine, format, f"fast_sweep_q{qubit}")
    except:
        data_fast = DataUnits(quantities={"frequency": "Hz"})
    try:
        data_precision = DataUnits.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
    except:
        data_precision = DataUnits(quantities={"frequency": "Hz"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
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

    datasets_fast = []
    copy = data_fast.df.copy()
    for i in range(len(copy)):
        datasets_fast.append(copy.drop_duplicates("frequency"))
        copy.drop(datasets_fast[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets_fast[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets_fast[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                opacity=0.3,
                name="Fast sweep MSR",
                showlegend=not bool(i),
                legendgroup="group1",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=datasets_fast[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets_fast[-1]["phase"].pint.to("rad").pint.magnitude,
                marker_color="rgb(204, 102, 102)",
                name="Fast sweep phase",
                opacity=0.3,
                showlegend=not bool(i),
                legendgroup="group2",
            ),
            row=1,
            col=2,
        )

    datasets_precision = []
    copy = data_precision.df.copy()
    for i in range(len(copy)):
        datasets_precision.append(copy.drop_duplicates("frequency"))
        copy.drop(datasets_precision[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets_precision[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets_precision[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(255, 130, 67)",
                opacity=0.3,
                name="Precision sweep MSR",
                showlegend=not bool(i),
                legendgroup="group3",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=datasets_precision[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets_precision[-1]["phase"].pint.to("rad").pint.magnitude,
                marker_color="rgb(104,40,96)",
                name="Precision sweep phase",
                opacity=0.3,
                showlegend=not bool(i),
                legendgroup="group4",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=data_fast.df.frequency.drop_duplicates().pint.to("GHz").pint.magnitude,
            y=data_fast.df.groupby("frequency")["MSR"]
            .mean()
            .pint.magnitude,  # CHANGE THIS TO
            name="average MSR fast sweep",
            marker_color="rgb(100, 0, 255)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_fast.df.frequency.drop_duplicates().pint.to("GHz").pint.magnitude,
            y=data_fast.df.groupby("frequency")["phase"].mean().pint.magnitude,
            name="average phase fast sweep",
            marker_color="rgb(204, 102, 102)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data_precision.df.frequency.drop_duplicates()
            .pint.to("GHz")
            .pint.magnitude,
            y=data_precision.df.groupby("frequency")["MSR"].mean().pint.magnitude,
            name="average MSR precision sweep",
            marker_color="rgb(255, 130, 67)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.df.frequency.drop_duplicates()
            .pint.to("GHz")
            .pint.magnitude,
            y=data_precision.df.groupby("frequency")["phase"].mean().pint.magnitude,
            name="average phase precision sweep",
            marker_color="rgb(104,40,96)",
        ),
        row=1,
        col=2,
    )

    if len(data_fast) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_fast.get_values("frequency", "GHz")),
            max(data_fast.get_values("frequency", "GHz")),
            2 * len(data_fast),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fitted MSR",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(102, 180, 71)",
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
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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
    print(smalldf.index)
    test = pd.concat(datasets)
    X, Y = [], []
    for i in smalldf.index:
        print(i)
        X.append(test["MSR"][i].mean().magnitude)
        Y.append(test["phase"][i].mean().magnitude)

    print(X, Y)
    fig.add_trace(
        go.Scatter(x=X, y=Y, name="mean"),
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


def frequency_flux_msr_phase(folder, routine, qubit, format):
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
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("current", "A"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("current", "A"),
            z=data.get_values("phase", "rad"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Current (A)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Current (A)",
    )
    return fig


def frequency_flux_msr_phase__matrix(folder, routine, qubit, format):
    fluxes = []
    for i in range(25):  # FIXME: 25 is hardcoded
        file = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        if os.path.exists(file):
            fluxes += [i]

    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes)
    fig = make_subplots(
        rows=2,
        cols=nb,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Frequency (Hz)",
        y_title="Current (A)",
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for j in fluxes:
        if j == fluxes[-1]:
            showscale = True
        else:
            showscale = False
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}_f{j}")
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("MSR", "V"),
                showscale=showscale,
            ),
            row=1,
            col=j,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("phase", "rad"),
                showscale=showscale,
            ),
            row=2,
            col=j,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig


def frequency_attenuation_msr_phase(folder, routine, qubit, format):
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
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("phase", "rad"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
    )
    return fig


def dispersive_frequency_msr_phase(folder, routine, qubit, formato):

    try:
        data_spec = DataUnits.load_data(folder, routine, formato, f"data_q{qubit}")
    except:
        data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})

    try:
        data_shifted = DataUnits.load_data(
            folder, routine, formato, f"data_shifted_q{qubit}"
        )
    except:
        data_shifted = DataUnits(
            name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
        )

    try:
        data_fit = Data.load_data(folder, routine, formato, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

    try:
        data_fit_shifted = Data.load_data(
            folder, routine, formato, f"fit_shifted_q{qubit}"
        )
    except:
        data_fit_shifted = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
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
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("MSR", "uV"),
            name="Spectroscopy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("phase", "rad"),
            name="Spectroscopy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("MSR", "uV"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("phase", "rad"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=2,
    )

    # fitting traces
    if len(data_spec) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_spec.get_values("frequency", "GHz")),
            max(data_spec.get_values("frequency", "GHz")),
            2 * len(data_spec),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit spectroscopy",
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

    # fitting shifted  traces
    if len(data_shifted) > 0 and len(data_fit_shifted) > 0:
        freqrange = np.linspace(
            min(data_shifted.get_values("frequency", "GHz")),
            max(data_shifted.get_values("frequency", "GHz")),
            2 * len(data_shifted),
        )
        params = [i for i in list(data_fit_shifted.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit_shifted.get_values("popt0"),
                    data_fit_shifted.get_values("popt1"),
                    data_fit_shifted.get_values("popt2"),
                    data_fit_shifted.get_values("popt3"),
                ),
                name="Fit shifted spectroscopy",
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
                text=f"The estimated shifted {params[0]} is {data_fit_shifted.df[params[0]][0]:.1f} Hz.",
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
