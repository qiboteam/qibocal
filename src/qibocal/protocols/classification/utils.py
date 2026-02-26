import plotly.express as px
import plotly.graph_objects as go

COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 200, 0),
    "orange": (255, 140, 0),
    "purple": (160, 32, 240),
    "yellow": (255, 255, 0),
}


def make_transparent_colorscale(color: str):
    """Return a simple transparent-to-solid colorscale for a few base colors."""
    r, g, b = COLORS[color]
    return [
        (0.0, f"rgba({r},{g},{b},0)"),  # transparent
        (1.0, f"rgba({r},{g},{b},1)"),  # opaque
    ]


def colors():
    """Yield one transparent colorscale at a time."""
    yield from COLORS


def plot_distribution(fig, data: dict, color: str, label: str):
    fig.add_trace(
        go.Histogram2dContour(
            x=data["I"],
            y=data["Q"],
            colorscale=make_transparent_colorscale(color),
            ncontours=20,
            showscale=False,
            legendgroup=label,
            showlegend=True,
            line=dict(width=0),
            name=label,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=data["I"],
            marker_color=color,
            legendgroup=label,
            showlegend=False,
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            y=data["Q"],
            marker_color=color,
            legendgroup=label,
            showlegend=False,
            opacity=0.5,
        ),
        row=2,
        col=2,
    )


def plot_confusion_matrix(confusion_matrix: list, labels: list):
    matrix = px.imshow(
        confusion_matrix,
        x=labels,
        y=labels,
        aspect="auto",
        text_auto=True,
        color_continuous_scale="Mint",
        title="Confusion matrix",
    )
    matrix.update_xaxes(title="Predicted State")
    matrix.update_yaxes(title="Actual State")
    return matrix
