import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import freq_r_mathieu, freq_r_transmon, line, lorenzian
from qibocal.plots.utils import get_color, get_data_subfolders


# Signals
def signals(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "State 0",
            "State 1",
        ),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(options=["qubit", "sample", "state"])

        states = data.df["state"].unique()
        # MSR = data.df["i"].pint.to("uV").pint.magnitude.unique()
        opacity = 1
        for state in states:
            state_data = data.df[data.df["state"] == state]
            fig.add_trace(
                go.Scatter(
                    x=state_data["sample"],
                    y=state_data["MSR"].pint.to("uV").pint.magnitude,
                    marker_color=get_color(2 * report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=not bool(state),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1 + state,
            )
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


# Integration weights calculation
def signal_0_1(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("State 0-1",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(options=["qubit", "sample", "state"])

        state = "1-0"
        opacity = 1

        state_data = data.df[data.df["state"] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data["sample"],
                y=state_data["weights"].pint.to("dimensionless").pint.magnitude,
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(state),
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=state_data["sample"],
                y=np.ones(
                    [len(state_data["weights"].pint.to("dimensionless").pint.magnitude)]
                ),
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(state),
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )

    report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
