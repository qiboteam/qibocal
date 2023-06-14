from typing import Iterable

import numpy as np
import plotly.graph_objects as go

from .utils import extract_from_data


def rb_figure(data, model, fit_label="", signal_label="signal", error_y=None):
    x, y = extract_from_data(data, signal_label, "depth", "mean")

    fig = go.Figure()

    # All samples
    fig.add_trace(
        go.Scatter(
            x=data.depth.tolist(),
            y=data.get(signal_label).tolist(),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="itertarions",
        )
    )

    # Averages
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )

    # If error_y is given, plot the error bars
    error_y_dict = None
    if error_y is not None:
        # Constant error bars
        if isinstance(error_y, Iterable) is False:
            error_y_dict = {"type": "constant", "value": error_y}
        # Symmetric error bars
        elif isinstance(error_y[0], Iterable) is False:
            error_y_dict = {"type": "data", "array": error_y}
        # Asymmetric error bars
        else:
            error_y_dict = {
                "type": "data",
                "symmetric": False,
                "array": error_y[1],
                "arrayminus": error_y[0],
            }
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=error_y_dict,
                line={"color": "#aa6464"},
                mode="markers",
                name="error bars",
            )
        )
    x_fit = np.linspace(min(x), max(x), len(x) * 20)
    y_fit = model(x_fit)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=fit_label,
            line=go.scatter.Line(dash="dot", color="#00cc96"),
        )
    )
    return fig


def carousel(fig_list: list):
    """Generate a Plotly figure as a carousel with a slider from a list of figures.

    Args:
        fig_list (List[plotly.graph_objects.Figure]): list of ``Plotly`` figures.

    Returns:
        :class:`plotly.graph_objects.Figure`: Carousel figure with a slider.
    """

    carousel_fig = go.Figure()
    steps = []
    fig_sizes = [len(fig.data) for fig in fig_list]
    for count, fig in enumerate(fig_list):
        for plot in fig.data:
            carousel_fig.add_trace(plot)
            carousel_fig.data[-1].visible = count == 0
        subplot_title = fig.layout.title.text if fig.layout.title.text else count + 1

        # Update slider data
        step = dict(
            label=subplot_title,
            method="update",
            args=[
                {"visible": [False] * sum(fig_sizes)},
                {"title": subplot_title},
            ],
        )

        # Toggle kth figure traces to "visible"
        step["args"][0]["visible"][
            len(carousel_fig.data) - fig_sizes[count] : len(carousel_fig.data)
        ] = [True] * fig_sizes[count]
        steps.append(step)

    # Create the slider
    sliders = [
        dict(
            active=0,
            pad={"t": 50},
            steps=steps,
        )
    ]

    carousel_fig.update_layout(sliders=sliders)
    return carousel_fig
