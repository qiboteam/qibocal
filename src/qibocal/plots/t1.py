import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import exp
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# T1
def t1_time_msr(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"wait": "ns"},
                options=["qubit", "iteration"],
            )

        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "label1",
                    "qubit",
                ]
            )

        data.df = data.df.drop(columns=["i", "q", "phase", "qubit"])
        iterations = data.df["iteration"].unique()
        waits = data.df["wait"].unique()

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1

        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["wait"],
                    y=iteration_data["MSR"] * 1e6,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )

        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
            fig.add_trace(
                go.Scatter(
                    x=waits,  # unique_waits,
                    y=data.df.groupby("wait")["MSR"].mean()
                    * 1e6,  # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

        # # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            waitrange = np.linspace(
                min(data.df["wait"]),
                max(data.df["wait"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=waitrange,
                    y=exp(
                        waitrange,
                        data_fit.df["popt0"][0],
                        data_fit.df["popt1"][0],
                        data_fit.df["popt2"][0],
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} | t1: {params['T1']:,.0f} ns.<br><br>"
            )

        report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
