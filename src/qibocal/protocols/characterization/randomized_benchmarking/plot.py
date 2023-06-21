from typing import Iterable, Optional, Union

import numpy as np
import plotly.graph_objects as go

from .utils import extract_from_data


def rb_figure(
    data,
    model,
    fit_label: Optional[str] = "",
    signal_label: Optional[str] = "signal",
    error_bars: Optional[Union[float, list, np.ndarray]] = None,
    **kwargs
):
    """Create Figure with RB signal, average values and fitting result.

    Args:
        data (:class:`RBData`): data of a protocol.
        model (callable): function that maps 1d array of ``x`` to ``y`` of the fitting result.
        fit_label (str, optional): label of the fit model in the plot legend.
        signal_label (str, optonal): name of the signal parameter in ``data``.
        error_bars (float or list or np.ndarray, optonal): error bars for the averaged signal.
        **kwargs: passed to the resulting figure's layout.

    Returns:
        plotly.graph_objects.Figure: resulting RB figure.
    """
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

    # If error_bars is given, plot the error bars
    error_bars_dict = None
    if error_bars is not None:
        # Constant error bars
        if isinstance(error_bars, Iterable) is False:
            error_bars_dict = {"type": "constant", "value": error_bars}
        # Symmetric error bars
        elif isinstance(error_bars[0], Iterable) is False:
            error_bars_dict = {"type": "data", "array": error_bars}
        # Asymmetric error bars
        else:
            error_bars_dict = {
                "type": "data",
                "symmetric": False,
                "array": error_bars[1],
                "arrayminus": error_bars[0],
            }
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=error_bars_dict,
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
    fig.update_layout(**kwargs)
    return fig


def carousel(fig_list: list):
    """Generate a Plotly figure as a carousel with a slider from a list of figures.

    Args:
        fig_list (List[plotly.graph_objects.Figure]): list of ``Plotly`` figures.
            Resulting figure will have the layout of ``fig_list[0]``, the slider values will
            correspond to figures' titles if given.

    Returns:
        :class:`plotly.graph_objects.Figure`: Carousel figure with a slider.
    """

    # Create a figure with the layout of `fig_list[0]` figure.
    carousel_fig = go.Figure()
    carousel_fig.update_layout(**fig_list[0].to_dict().get("layout", {}))

    # Record each figure as a step for the slider
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
            currentvalue={"prefix": "Irrep "},
            pad={"t": 50},
            steps=steps,
        )
    ]
    carousel_fig.update_layout(sliders=sliders)
    return carousel_fig
