import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import ramsey
from qibocal.plots.utils import get_color, get_data_subfolders


# Ramsey oscillations
def time_msr(folder, routine, qubit, format):
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
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
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "label1",
                    "label2",
                    "label3",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        waits = data.df["wait"].pint.to("ns").pint.magnitude.unique()

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["wait"].pint.to("ns").pint.magnitude,
                    y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
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
            fig.add_trace(
                go.Scatter(
                    x=waits,
                    y=data.df.groupby("wait")["MSR"]
                    .mean()
                    .pint.to("uV")
                    .pint.magnitude,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            waitrange = np.linspace(
                min(data.get_values("wait", "ns")),
                max(data.get_values("wait", "ns")),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=waitrange,
                    y=ramsey(
                        waitrange,
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                        data_fit.get_values("popt4"),
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | delta_frequency: {params['delta_frequency']:,.0f} Hz<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | corrected_qubit_frequency: {params['corrected_qubit_frequency']:,.0f} Hz<br>"
                )
                + (f"q{qubit}/r{report_n} | t2: {params['t2']:,.0f} ns.<br><br>")
            )
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
