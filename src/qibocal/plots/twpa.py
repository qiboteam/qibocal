import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import freq_r_mathieu, freq_r_transmon, line, lorenzian
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# Punchout
def twpa_frequency(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "twpa_frequency": "Hz"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].unique()
        twpa_frequencys = data.df["twpa_frequency"].unique()
        averaged_data = (
            data.df.drop(columns=["i", "q", "qubit", "iteration"])
            .groupby(["frequency", "twpa_frequency"], as_index=False)
            .mean()
        )

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"],
                y=averaged_data["twpa_frequency"],
                z=averaged_data["MSR"],
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
        )
        fig.update_yaxes(title_text="TWPA frequency (Hz)", row=1 + report_n, col=1)
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"],
                y=averaged_data["twpa_frequency"],
                z=averaged_data["phase"],
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
        )
        fig.update_yaxes(title_text="twpa_frequency (Hz)", row=1 + report_n, col=2)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report


# Punchout
def twpa_power(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "twpa_power": "dB"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].unique()
        twpa_powers = data.df["twpa_power"].unique()
        averaged_data = (
            data.df.drop(columns=["i", "q", "qubit", "iteration"])
            .groupby(["frequency", "twpa_power"], as_index=False)
            .mean()
        )

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"],
                y=averaged_data["twpa_power"],
                z=averaged_data["MSR"],
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
        )
        fig.update_yaxes(title_text="TWPA power (dB)", row=1 + report_n, col=1)
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"],
                y=averaged_data["twpa_power"],
                z=averaged_data["phase"],
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
        )
        fig.update_yaxes(title_text="twpa_power (dB)", row=1 + report_n, col=2)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report
