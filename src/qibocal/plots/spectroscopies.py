import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import lorenzian
from qibocal.plots.utils import get_data_subfolders


def frequency_msr_phase(folder, routine, qubit, format):

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
    j = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_fast = DataUnits.load_data(
                folder, subfolder, routine, format, f"fast_sweep"
            )
            data_fast.df = data_fast.df[
                data_fast.df["qubit"] == int(qubit)
            ].reset_index(drop=True)
        except:
            data_fast = DataUnits(quantities={"frequency": "Hz"}, options=["qubit"])
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
                    "label1",
                    "label2",
                ]
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
                    name=f"q{qubit}/r{j} MSR",
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
                    name=f"q{qubit}/r{j} phase",
                    opacity=0.3,
                    showlegend=not bool(i),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data_fast.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_fast.df.groupby("frequency")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"q{qubit}/r{j} avg MSR",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_fast.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_fast.df.groupby("frequency")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"q{qubit}/r{j} avg phase",
                marker_color="rgb(204, 102, 102)",
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
            params = [k for k in list(data_fit.df.keys()) if "popt" not in k]
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
                    name=f"q{qubit}/r{j} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color="rgb(102, 180, 71)",
                ),
                row=1,
                col=1,
            )
            if data_fit.df[params[2]][0] == 0:
                param2 = ""
            else:
                param2 = f"q{qubit}/r{j} {params[2]}: {data_fit.df[params[2]][0]:.1f} Hz.<br>"

            fitting_report = fitting_report + (
                f"{param2}q{qubit}/r{j} {params[1]}: {data_fit.df[params[1]][0]:.3f} uV.<br>q{qubit}/r{j} {params[0]}: {data_fit.df[params[0]][0]:.0f} Hz.<br><br>"
            )

        j += 1

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
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Resonator and qubit spectroscopies
def frequency_attenuation_msr_phase__cut(folder, routine, qubit, format):

    plot1d_attenuation = 56  # attenuation value to use for 1D frequency vs MSR plot

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    i = 0
    for subfolder in subfolders:

        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"frequency": "Hz"})

        # index data on a specific attenuation value
        smalldf = data.df[
            data.get_values("attenuation", "dB") == plot1d_attenuation
        ].copy()
        smalldf1 = smalldf.copy()
        # split multiple software averages to different datasets
        datasets = []
        for j in range(len(smalldf)):
            datasets.append(smalldf.drop_duplicates("frequency"))
            smalldf.drop(datasets[-1].index, inplace=True)

            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("V").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name="MSR",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
            )

        fig.add_trace(
            go.Scatter(
                x=smalldf1.frequency.drop_duplicates().pint.to("GHz").pint.magnitude,
                y=smalldf1.groupby("frequency")["MSR"].mean().pint.magnitude,
                name="average MSR",
                marker_color="rgb(100, 0, 255)",
            ),
        )

        i += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


# Punchout
def frequency_attenuation_msr_phase(folder, routine, qubit, format):

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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"frequency": "Hz", "attenuation": "dB"})

        size = len(data.df.attenuation.drop_duplicates()) * len(
            data.df.frequency.drop_duplicates()  # pylint: disable=E1101
        )

        fig.add_trace(
            go.Heatmap(
                x=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .frequency.mean()
                .pint.to("GHz")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .attenuation.mean()
                .pint.to("dB")
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
                .frequency.mean()
                .pint.to("GHz")
                .pint.magnitude,
                y=data.df.groupby(data.df.index % size)  # pylint: disable=E1101
                .attenuation.mean()
                .pint.to("dB")
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
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"]["yaxis"]["title"] = "Attenuation (dB)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"

        i += 1

    return fig


# Dispersive shift
def dispersive_frequency_msr_phase(folder, routine, qubit, format):

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
            data_spec = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data_spec.df = data_spec.df[
                data_spec.df["qubit"] == int(qubit)
            ].reset_index(drop=True)
        except:
            data_spec = DataUnits(name=f"data", quantities={"frequency": "Hz"})

        try:
            data_shifted = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_shifted"
            )
            data_shifted.df = data_shifted.df[
                data_shifted.df["qubit"] == int(qubit)
            ].reset_index(drop=True)
        except:
            data_shifted = DataUnits(
                name=f"data_shifted", quantities={"frequency": "Hz"}
            )

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
                    "label1",
                    "label2",
                ]
            )

        try:
            data_fit_shifted = Data.load_data(
                folder, subfolder, routine, format, f"fit_shifted_q{qubit}"
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

        datasets_spec = []
        copy = data_spec.df.copy()
        for j in range(len(copy)):
            datasets_spec.append(copy.drop_duplicates("frequency"))
            copy.drop(datasets_spec[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets_spec[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets_spec[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"q{qubit}/r{i} MSR",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets_spec[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets_spec[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"q{qubit}/r{i} phase",
                    opacity=0.3,
                    showlegend=not bool(j),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data_spec.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_spec.df.groupby("frequency")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"q{qubit}/r{i} average MSR",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_spec.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_spec.df.groupby("frequency")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"q{qubit}/r{i} average phase",
                marker_color="rgb(104, 40, 96)",
            ),
            row=1,
            col=2,
        )

        datasets_shifted = []
        copy = data_shifted.df.copy()
        for j in range(len(copy)):
            datasets_shifted.append(copy.drop_duplicates("frequency"))
            copy.drop(datasets_shifted[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets_shifted[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets_shifted[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"q{qubit}/r{i} shifted MSR",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets_shifted[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets_shifted[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"q{qubit}/r{i} shifted phase",
                    opacity=0.3,
                    showlegend=not bool(j),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data_shifted.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_shifted.df.groupby("frequency")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"q{qubit}/r{i} average shifted MSR",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_shifted.df.frequency.drop_duplicates()  # pylint: disable=E1101
                .pint.to("GHz")
                .pint.magnitude,
                y=data_shifted.df.groupby("frequency")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"q{qubit}/r{i} average shifted phase",
                marker_color="rgb(104, 40, 96)",
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
                    name=f"q{qubit}/r{i} Fit spectroscopy",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[2]}: {data_fit.df[params[2]][0]:.1f} Hz.<br>"
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
                    name=f"q{qubit}/r{i} Fit shifted spectroscopy",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} shifted {params[2]}: {data_fit_shifted.df[params[2]][0]:.1f} Hz.<br><br>"
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
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig
