import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import flipping
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# Flipping
def flips_msr(folder, routine, qubit, format):
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
                quantities={"flips": "dimensionless"}, options=["qubit", "iteration"]
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
                    "popt3",
                    "popt4",
                    "label1",
                    "label2",
                    "qubit",
                ]
            )

        data.df = data.df.drop(columns=["i", "q", "phase", "qubit"])
        iterations = data.df["iteration"].unique()
        flips = data.df["flips"].unique()

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["flips"],
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
                    x=flips,
                    y=data.df.groupby("flips")["MSR"].mean()  # pylint: disable=E1101
                    * 1e6,
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
            flips_range = np.linspace(
                min(data.df["flips"]),
                max(data.df["flips"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=flips_range,
                    y=flipping(
                        flips_range,
                        float(data_fit.df["popt0"]),
                        float(data_fit.df["popt1"]),
                        float(data_fit.df["popt2"]),
                        float(data_fit.df["popt3"]),
                    ),
                    name=f"q{qubit}/r{report_n}: Fit MSR",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} | amplitude_correction_factor: {params['amplitude_correction_factor']:.4f}<br>"
                + f"q{qubit}/r{report_n} | corrected_amplitude: {params['corrected_amplitude']:.4f}<br><br>"
            )

        report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
