from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import exp
from qibocal.plots.utils import get_color, get_data_subfolders


# T2
def time_msr(folder, routine, qubit, format):
    fig = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
            iterations = data.df["iteration"].unique()
        except:
            data = DataUnits(
                name=f"data",
                quantities={"wait": "ns", "t_max": "ns"},
                options=["qubit", "iteration"],
            )
        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "qubit",
                    "t2",
                    "corrected_qubit_frequency",
                    "delta_frequency",
                    "popt4",
                    "popt3",
                    "popt2",
                    "popt1",
                    "popt0",
                    "timestamp",
                ]
            )

        # TODO: average over iterations
        for iteration in iterations:
            pass

        # plot raw data
        iteration_data = data.df[data.df["iteration"] == 0]
        fig.add_trace(
            go.Heatmap(
                x=iteration_data["wait"].pint.to("ns").pint.magnitude,
                z=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                y=iteration_data["timestamp"],
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )
        # add fitting trace
        if len(data_fit) > 0 and (qubit in data_fit.df["qubit"].values):
            fig.add_trace(
                go.Scatter(
                    x=data_fit.df["timestamp"],  # "%Y-%m-%d %H:%M:%S.%f" ,
                    y=data_fit.df["t2"],
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=2,
                col=1,
            )

        report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="Date",
        xaxis2_title="Date",
        yaxis2_title="t2 (ns)",
    )
    return fig
