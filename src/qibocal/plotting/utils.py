from typing import Optional

import plotly.graph_objects as go


def scatter_plot(
    x: list, y: list, label: str, error_y: Optional[list] = None
) -> go.Scatter:
    """Creates basic scatter plot with error bars."""
    # TODO: check if with error_y None we get something that makes sense.
    return go.Scatter(
        x=x,
        y=y,
        error_y=dict(
            type="data",
            array=error_y,
            visible=True,
        ),
        showlegend=True,
        name=label,
        legendgroup=label,
        mode="markers",
    )


def fit_plot(x: list, y: list, label: str) -> go.Scatter:
    """Create basic line plot."""
    return go.Scatter(
        x=x,
        y=y,
        name=label,
    )
