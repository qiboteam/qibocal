"""Plotting helpers for qibocal protocols."""

from colorsys import hls_to_rgb

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.stats import norm as scipy_norm

from qibocal.auto.operation import Data, QubitId, Results
from qibocal.fitting.classifier import run

from .constants import (
    COLUMNWIDTH,
    LEGEND_FONT_SIZE,
    MARGIN,
    MESH_SIZE,
    SPACING,
    TITLE_SIZE,
)


def get_color_state0(number) -> str:
    return "rgb" + str(hls_to_rgb((-0.35 - number * 9 / 20) % 1, 0.6, 0.75))


def get_color_state1(number) -> str:
    return "rgb" + str(hls_to_rgb((-0.02 - number * 9 / 20) % 1, 0.6, 0.75))


def evaluate_grid(
    data: NDArray,
):
    """
    This function returns a matrix grid evaluated from
    the datapoints `data`.
    """
    max_x = (
        max(
            0,
            data["i"].max(),
        )
        + MARGIN
    )
    max_y = (
        max(
            0,
            data["q"].max(),
        )
        + MARGIN
    )
    min_x = (
        min(
            0,
            data["i"].min(),
        )
        - MARGIN
    )
    min_y = (
        min(
            0,
            data["q"].min(),
        )
        - MARGIN
    )
    i_values, q_values = np.meshgrid(
        np.linspace(min_x, max_x, num=MESH_SIZE),
        np.linspace(min_y, max_y, num=MESH_SIZE),
    )
    return np.vstack([i_values.ravel(), q_values.ravel()]).T


def plot_results(
    data: Data, qubit: QubitId, qubit_states: list, fit: Results
) -> list[go.Figure]:
    """
    Plots for the qubit and qutrit classification.

    Args:
        data (Data): acquisition data
        qubit (QubitID): qubit
        qubit_states (list): list of qubit states available.
        fit (Results): fit results
    """
    figures = []
    models_name = data.classifiers_list
    qubit_data = data.data[qubit]
    grid = evaluate_grid(qubit_data)

    fig = make_subplots(
        rows=2,
        cols=len(models_name),
        horizontal_spacing=SPACING * 3 / len(models_name) * 3,
        vertical_spacing=SPACING,
        subplot_titles=[run.pretty_name(model) for model in models_name],
        column_width=[COLUMNWIDTH] * len(models_name),
    )

    for i, model in enumerate(models_name):
        if fit is not None:
            predictions = fit.grid_preds[qubit][i]
            fig.add_trace(
                go.Contour(
                    x=grid[:, 0],
                    y=grid[:, 1],
                    z=np.array(predictions).flatten(),
                    showscale=False,
                    colorscale=[get_color_state0(i), get_color_state1(i)],
                    opacity=0.2,
                    name="Score",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=i + 1,
            )

        model = run.pretty_name(model)
        max_x = max(grid[:, 0])
        max_y = max(grid[:, 1])
        min_x = min(grid[:, 0])
        min_y = min(grid[:, 1])

        # Colorset for plots
        COLORS = px.colors.qualitative.Plotly[0:qubit_states]
        if COLORS[0].startswith("#"):
            COLORS = [
                f"rgba({int(COLORS[j][1:3], 16)},{int(COLORS[j][3:5], 16)},{int(COLORS[j][5:7], 16)},0.5)"
                for j in range(len(COLORS))
            ]

        for state in range(qubit_states):
            state_data = qubit_data[qubit_data["state"] == state]

            fig.add_trace(
                go.Scatter(
                    x=state_data["i"],
                    y=state_data["q"],
                    name=f"{model}: state {state}",
                    legendgroup=f"{model}: state {state}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker={"size": 3},
                    marker_color=COLORS[state],
                ),
                row=1,
                col=i + 1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[np.average(state_data["i"])],
                    y=[np.average(state_data["q"])],
                    name=f"{model}: state {state}",
                    legendgroup=f"{model}: state {state}",
                    showlegend=False,
                    mode="markers",
                    marker={"size": 10},
                ),
                row=1,
                col=i + 1,
            )

            # Add 1D histogram trace rotated by rot_angle from the fit results
            if fit is not None and getattr(fit, "rotation_angle", None) is not None:
                rot_angle = np.round(fit.rotation_angle[qubit], 3)
                threshold = np.round(fit.threshold[qubit], 3)

                x, y = state_data["i"], state_data["q"]
                c, s = np.cos(rot_angle), np.sin(rot_angle)
                rot = np.array([[c, -s], [s, c]])
                rotated = np.vstack([x, y]).T @ rot.T
                rotated[:, 0] = rotated[:, 0]

                # histogram using only the x values
                hist, bin_edges = np.histogram(
                    rotated[:, 0],
                    bins=30,
                    density=True,
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Gaussian fit to histogram
                mu, std = scipy_norm.fit(rotated[:, 0])
                pdf = scipy_norm.pdf(bin_centers, mu, std)

                fig.add_trace(
                    go.Bar(
                        x=bin_centers - threshold,
                        y=hist,
                        name=f"{model}: state {state} hist",
                        legendgroup=f"{model}: state {state}",
                        showlegend=False,
                        marker={"color": COLORS[state]},
                        width=(bin_centers[1] - bin_centers[0])
                        if len(bin_centers) > 1
                        else 0.1,
                    ),
                    row=2,
                    col=i + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers - threshold,
                        y=pdf,
                        name=f"{model}: state {state} norm fit",
                        mode="lines",
                        legendgroup=f"{model}: state {state}",
                        showlegend=False,
                        line={"width": 2, "color": COLORS[state]},
                    ),
                    row=2,
                    col=i + 1,
                )

                # Add vertical line for threshold
                fig.add_trace(
                    go.Scatter(
                        x=[0, 0],
                        y=[0, max(hist) * 1.1],
                        name="threshold",  # No name for legend
                        mode="lines",
                        line={"color": "black", "width": 2, "dash": "dot"},
                        showlegend=False,
                    ),
                    row=2,
                    col=i + 1,
                )

            fig.update_xaxes(
                title_text="i [a.u.]",
                range=[min_x, max_x],
                row=1,
                col=i + 1,
                autorange=False,
                rangeslider={"visible": False},
            )
            fig.update_yaxes(
                title_text="q [a.u.]",
                range=[min_y, max_y],
                scaleanchor="x",
                scaleratio=1,
                row=1,
                col=i + 1,
            )

    fig.update_layout(
        autosize=False,
        height=COLUMNWIDTH,
        width=COLUMNWIDTH * len(models_name),
        title={"text": "Results", "font": {"size": TITLE_SIZE}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "left",
            "y": -0.3,
            "x": 0,
            "itemsizing": "constant",
            "font": {"size": LEGEND_FONT_SIZE},
        },
    )
    figures.append(fig)

    if fit is not None and len(models_name) != 1:
        fig_benchmarks = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=SPACING,
            vertical_spacing=SPACING,
            subplot_titles=(
                "accuracy",
                "testing time [s]",
                "training time [s]",
            ),
        )
        for i, model in enumerate(models_name):
            for plot in range(3):
                fig_benchmarks.add_trace(
                    go.Scatter(
                        x=[model],
                        y=[fit.benchmark_table[qubit][i][plot]],
                        mode="markers",
                        showlegend=False,
                        marker={"size": 10, "color": get_color_state1(i)},
                    ),
                    row=1,
                    col=plot + 1,
                )

        fig_benchmarks.update_yaxes(type="log", row=1, col=2)
        fig_benchmarks.update_yaxes(type="log", row=1, col=3)
        fig_benchmarks.update_layout(
            autosize=False,
            height=COLUMNWIDTH,
            width=COLUMNWIDTH * 3,
            title={"text": "Benchmarks", "font": {"size": TITLE_SIZE}},
        )

        figures.append(fig_benchmarks)
    return figures
