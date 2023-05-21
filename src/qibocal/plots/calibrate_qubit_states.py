import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import (
    get_color_state0,
    get_color_state1,
    get_data_subfolders,
    load_data,
)


# For calibrate qubit states
def qubit_states(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        # subplot_titles=("Calibrate qubit states"),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    max_x, max_y, min_x, min_y = 0, 0, 0, 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(options=["qubit", "iteration", "state"])

        try:
            parameters = load_data(folder, subfolder, routine, format, "parameters")
            parameters.df = parameters.df[parameters.df["qubit"] == qubit]

            average_state0 = complex(parameters.df.iloc[0]["average_state0"])
            average_state1 = complex(
                parameters.df.iloc[0]["average_state1"]  # pylint: disable=E1101
            )
            rotation_angle = parameters.df.iloc[0][  # pylint: disable=E1101
                "rotation_angle"
            ]  # pylint: disable=E1101
            threshold = parameters.df.iloc[0]["threshold"]  # pylint: disable=E1101
            fidelity = parameters.df.iloc[0]["fidelity"]  # pylint: disable=E1101
            assignment_fidelity = parameters.df.iloc[0][
                "assignment_fidelity"
            ]  # pylint: disable=E1101

        except:
            parameters = Data(
                name=f"parameters",
                quantities=[
                    "rotation_angle",  # in rad
                    "threshold",
                    "fidelity",
                    "assignment_fidelity",
                    "average_state0",
                    "average_state1",
                    "qubit",
                ],
            )

        state0_data = data.df[data.df["state"] == 0]
        state1_data = data.df[data.df["state"] == 1]

        fig.add_trace(
            go.Scatter(
                x=state0_data["i"],
                y=state0_data["q"],
                name=f"q{qubit}/r{report_n}: state 0",
                legendgroup=f"q{qubit}/r{report_n}: state 0",
                mode="markers",
                showlegend=False,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state0(report_n)),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=state1_data["i"],
                y=state1_data["q"],
                name=f"q{qubit}/r{report_n}: state 1",
                legendgroup=f"q{qubit}/r{report_n}: state 1",
                mode="markers",
                showlegend=False,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state1(report_n)),
            ),
            row=1,
            col=1,
        )

        max_x = max(
            max_x,
            state0_data["i"].max(),
            state1_data["i"].max(),
        )
        max_y = max(
            max_y,
            state0_data["q"].max(),
            state1_data["q"].max(),
        )
        min_x = min(
            min_x,
            state0_data["i"].min(),
            state1_data["i"].min(),
        )
        min_y = min(
            min_y,
            state0_data["q"].min(),
            state1_data["q"].min(),
        )

        fig.add_trace(
            go.Scatter(
                x=[average_state0.real],
                y=[average_state0.imag],
                name=f"q{qubit}/r{report_n}: state 0",
                legendgroup=f"q{qubit}/r{report_n}: state 0",
                showlegend=True,
                mode="markers",
                marker=dict(size=10, color=get_color_state0(report_n)),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[average_state1.real],
                y=[average_state1.imag],
                name=f"q{qubit}/r{report_n}: state 1",
                legendgroup=f"q{qubit}/r{report_n}: state 1",
                showlegend=True,
                mode="markers",
                marker=dict(size=10, color=get_color_state1(report_n)),
            ),
            row=1,
            col=1,
        )

        title_text = ""
        title_text += (
            f"q{qubit}/r{report_n} | average state 0: ({average_state0:.6f})<br>"
        )
        title_text += (
            f"q{qubit}/r{report_n} | average state 1: ({average_state1:.6f})<br>"
        )
        title_text += f"q{qubit}/r{report_n} | rotation angle: {rotation_angle:.3f}<br>"
        title_text += f"q{qubit}/r{report_n} | threshold: {threshold:.6f}<br>"
        title_text += f"q{qubit}/r{report_n} | fidelity: {fidelity:.3f}<br>"
        title_text += (
            f"q{qubit}/r{report_n} | assignment fidelity: {assignment_fidelity:.3f}<br>"
        )

        fitting_report = fitting_report + title_text
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        xaxis_range=(min_x, max_x),
        yaxis_range=(min_y, max_y),
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    figures.append(fig)

    return figures, fitting_report
