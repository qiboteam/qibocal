
import lmfit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.auto.operation import Results
from qibocal.config import log
from qibocal.plots.utils import get_color


# Signals
def signals(data, fit, qubit):
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
    qubit_data = data.df[data.df["qubit"] == qubit].drop(columns=["i", "q", "qubit"])

    fitting_report = ""

    states = data.df["state"].unique()
    # MSR = data.df["i"].pint.to("uV").pint.magnitude.unique()
    opacity = 1
    for state in states:
        state_data = data.df[data.df["state"] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data["sample"],
                y=state_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(1),
                opacity=opacity,
            ),
            row=1,
            col=1 + state,
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
