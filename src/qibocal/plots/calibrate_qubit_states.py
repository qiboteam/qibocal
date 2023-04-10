import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import (
    get_color_state0,
    get_color_state1,
    get_data_subfolders,
    load_data,
)

LEGENDSIZE = 50
COLUMNWIDTH = 600
ROC_LENGHT = 800
ROC_WIDTH = 800
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1


# For calibrate qubit states
def qubit_states(folder, routine, qubit, format):
    figures = []

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""

    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(options=["qubit", "iteration", "state"])

        parameters = load_data(folder, subfolder, routine, format, "parameters")
        parameters.df = parameters.df[parameters.df["qubit"] == qubit]
        models_name = parameters.df["model_name"].to_list()

        state0_data = data.df[data.df["state"] == 0]
        state1_data = data.df[data.df["state"] == 1]

        grid = _bytes_to_np(parameters.df.iloc[0]["grid"], np.float64).reshape((-1, 2))

        accuracy = []
        training_time = []
        testing_time = []

        fig = make_subplots(
            rows=1,
            cols=len(models_name),
            horizontal_spacing=SPACING * 3 / len(models_name),
            vertical_spacing=SPACING,
            subplot_titles=(models_name),
            column_width=[COLUMNWIDTH] * len(models_name),
        )
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        fig_benchmarks = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=SPACING,
            vertical_spacing=SPACING,
            subplot_titles=("accuracy", "training time (s)", "testing time (s)"),
            # column_width = [COLUMNWIDTH]*3
        )

        for i, model in enumerate(models_name):
            y_test = _bytes_to_np(parameters.df.iloc[i]["y_test"], np.int64)
            y_pred = _bytes_to_np(parameters.df.iloc[i]["y_pred"], np.int64)
            predictions = _bytes_to_np(parameters.df.iloc[i]["predictions"], np.int64)
            # Evaluate the ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)

            name = f"{model} (AUC={auc_score:.2f})"
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=name,
                    mode="lines",
                    marker=dict(size=3, color=get_color_state0(report_n)),
                )
            )

            max_x = max(grid[:, 0])
            max_y = max(grid[:, 1])
            min_x = min(grid[:, 0])
            min_y = min(grid[:, 1])

            try:
                parameters = load_data(folder, subfolder, routine, format, "parameters")
                parameters.df = parameters.df[parameters.df["qubit"] == qubit]

                average_state0 = complex(parameters.df.iloc[i]["average_state0"])
                average_state1 = complex(
                    parameters.df.iloc[i]["average_state1"]  # pylint: disable=E1101
                )
                rotation_angle = parameters.df.iloc[i][  # pylint: disable=E1101
                    "rotation_angle"
                ]  # pylint: disable=E1101
                threshold = parameters.df.iloc[i]["threshold"]  # pylint: disable=E1101
                fidelity = parameters.df.iloc[i]["fidelity"]  # pylint: disable=E1101
                assignment_fidelity = parameters.df.iloc[i][
                    "assignment_fidelity"
                ]  # pylint: disable=E1101
                accuracy = parameters.df.iloc[i]["accuracy"]  # pylint: disable=E1101
                training_time = parameters.df.iloc[i]["training_time"]
                testing_time = parameters.df.iloc[i]["testing_time"]

            except:
                parameters = Data(
                    name=f"parameters",
                    quantities=[
                        "rotation_angle",
                        "threshold",
                        "fidelity",
                        "assignment_fidelity",
                        "average_state0",
                        "average_state1",
                        "qubit",
                    ],
                )

            fig_benchmarks.add_trace(
                go.Scatter(
                    x=[model],
                    y=[accuracy],
                    mode="markers",
                    showlegend=False,
                    # opacity=0.7,
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
                row=1,
                col=1,
            )

            fig_benchmarks.add_trace(
                go.Scatter(
                    x=[model],
                    y=[training_time],
                    mode="markers",
                    showlegend=False,
                    # opacity=0.7,
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
                row=1,
                col=2,
            )

            fig_benchmarks.add_trace(
                go.Scatter(
                    x=[model],
                    y=[testing_time],
                    mode="markers",
                    showlegend=False,
                    # opacity=0.7,
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Scatter(
                    x=state0_data["i"].to_list(),
                    y=state0_data["q"].to_list(),
                    name=f"q{qubit}/{model}: state 0",
                    legendgroup=f"q{qubit}/{model}: state 0",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state0(report_n)),
                ),
                row=1,
                col=report_n + 1,
            )

            fig.add_trace(
                go.Scatter(
                    x=state1_data["i"].to_list(),
                    y=state1_data["q"].to_list(),
                    name=f"q{qubit}/{model}: state 1",
                    legendgroup=f"q{qubit}/{model}: state 1",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state1(report_n)),
                ),
                row=1,
                col=report_n + 1,
            )

            fig.add_trace(
                go.Contour(
                    x=grid[:, 0],
                    y=grid[:, 1],
                    z=predictions,
                    showscale=False,
                    colorscale=[get_color_state0(report_n), get_color_state1(report_n)],
                    opacity=0.4,
                    name="Score",
                    hoverinfo="skip",
                ),
                row=1,
                col=report_n + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[average_state0.real],
                    y=[average_state0.imag],
                    name=f"q{qubit}/{model}: state 0",
                    legendgroup=f"q{qubit}/{model}: state 0",
                    showlegend=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state0(report_n)),
                ),
                row=1,
                col=report_n + 1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[average_state1.real],
                    y=[average_state1.imag],
                    name=f"q{qubit}/{model}: state 1",
                    legendgroup=f"q{qubit}/{model}: state 1",
                    showlegend=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
                row=1,
                col=report_n + 1,
            )
            fig.update_xaxes(
                title_text=f"i (V)",
                range=[min_x, max_x],
                row=1,
                col=report_n + 1,
                autorange=False,
                rangeslider=dict(visible=False),
            )
            fig.update_yaxes(
                title_text="q (V)",
                range=[min_y, max_y],
                scaleanchor="x",
                scaleratio=1,
                row=1,
                col=report_n + 1,
            )

            title_text = ""
            if models_name[i] == "qubit_fit":
                title_text += (
                    f"q{qubit}/{model} | average state 0: ({average_state0:.6f})<br>"
                )
                title_text += (
                    f"q{qubit}/{model} | average state 1: ({average_state1:.6f})<br>"
                )
                title_text += (
                    f"q{qubit}/{model} | rotation angle: {rotation_angle:.3f}<br>"
                )
                title_text += f"q{qubit}/{model} | threshold: {threshold:.6f}<br>"
                title_text += f"q{qubit}/{model} | fidelity: {fidelity:.3f}<br>"
                title_text += f"q{qubit}/{model} | assignment fidelity: {assignment_fidelity:.3f}<br>"

            fitting_report = fitting_report + title_text
            report_n += 1

            fig.update_layout(
                # showlegend=False,
                uirevision="0",  # ``uirevision`` allows zooming while live plotting
                autosize=False,
                height=COLUMNWIDTH,
                width=COLUMNWIDTH * len(models_name),
                title=dict(text="Results", font=dict(size=TITLE_SIZE)),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    xanchor="left",
                    y=-0.3,
                    x=0,
                    itemsizing="constant",
                    font=dict(size=LEGEND_FONT_SIZE),
                ),
            )
            fig_benchmarks.update_yaxes(type="log", row=1, col=2)
            fig_benchmarks.update_yaxes(type="log", row=1, col=3)
            fig_benchmarks.update_layout(
                autosize=False,
                height=COLUMNWIDTH,
                width=COLUMNWIDTH * 3,
                title=dict(text="Benchmarks", font=dict(size=TITLE_SIZE)),
            )
            fig_roc.update_layout(
                width=ROC_WIDTH,
                height=ROC_LENGHT,
                title=dict(text="ROC curves", font=dict(size=TITLE_SIZE)),
                legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
            )

    figures.append(fig_roc)
    figures.append(fig)
    figures.append(fig_benchmarks)
    return figures, fitting_report


def _bytes_to_np(data: bytes, type):
    # This function convert a bytes in numpy array
    return np.frombuffer(eval(data), dtype=type)
