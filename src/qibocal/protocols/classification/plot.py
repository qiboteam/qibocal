"""Plots for single-shot classification."""

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm as scipy_norm

from qibocal.auto.operation import Data, QubitId, Results
from qibocal.fitting.classifier import run

from ..utils import (
    LEGEND_FONT_SIZE,
    SPACING,
    evaluate_grid,
    get_color_state1,
)

__all__ = ["plot_results"]

STATE_COLORS = px.colors.qualitative.Plotly
"""Colors of the prepared states."""
BOUNDARY_COLORS = px.colors.qualitative.Dark2
"""Colors of the classifiers' decision boundaries."""
BOUNDARY_DASHES = ["dash", "dot", "dashdot", "longdash"]
"""Line styles of the classifiers' decision boundaries, distinguishing overlapping
ones."""
MARGIN = {"t": 30}
"""Figures' margins, with no title."""
HEIGHT = 600
"""Height of the IQ plane figure, in pixels."""
HISTOGRAM_HEIGHT = 400
"""Height of the projected histogram figure, in pixels."""
BINS_PER_STD = 3
"""Density bins per standard deviation of the narrowest state distribution."""
MIN_BINS = 20
MAX_BINS = 200
"""Bounds on the number of density bins per axis."""
HISTOGRAM_BINS = 60
"""Bins of the projected histogram."""
PADDING = 0.05
"""Fraction of the data extent left empty around the IQ plane plot."""
OPACITY_EXPONENT = 0.6
"""Exponent of the opacity in the fraction of the maximum count, cf. :func:`_density`.

A linear opacity (``1``) hides the tails of the distributions, while a logarithmic
one over-represents them, merging the cores of overlapping states.
"""


def _rgba(color: str, alpha: float) -> str:
    """Convert a hex color to an RGBA string."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _robust_std(values: npt.NDArray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def _edges(states: list[npt.NDArray]) -> npt.NDArray:
    """Square density bins, shared by all the states and both axes.

    The bin width is a fraction of the narrowest state distribution, such that each
    blob is resolved, independently of how far apart the blobs are.
    """
    points = np.concatenate(states)
    low, high = points.min(axis=0), points.max(axis=0)
    extent = float(np.max(high - low)) or 1.0
    spreads = [_robust_std(state[:, axis]) for state in states for axis in (0, 1)]
    spread = min((s for s in spreads if s > 0), default=extent)
    bins = int(np.clip(np.ceil(extent / spread * BINS_PER_STD), MIN_BINS, MAX_BINS))
    width = extent / bins
    return low + width * np.arange(bins + 1)[:, np.newaxis]


def _rgb(color: str) -> npt.NDArray:
    """Convert a hex color to RGB components."""
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _density(states: list[npt.NDArray], edges: npt.NDArray) -> go.Image:
    """Density of the shots of all the prepared states, as a single image.

    Each bin is colored with the average of the states' colors, weighted by their
    counts in the bin, such that overlapping states blend symmetrically, regardless of
    the order in which they are drawn. The opacity grows with the total count of the
    bin, as a power of its fraction of the maximum count, cf. :data:`OPACITY_EXPONENT`.
    Only the bin counts are stored in the figure, so its size does not depend on the
    number of shots.
    """
    counts = np.array(
        [
            np.histogram2d(points[:, 0], points[:, 1], bins=[edges[:, 0], edges[:, 1]])[
                0
            ].T
            for points in states
        ]
    )
    total = counts.sum(axis=0)
    weights = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    colors = np.array([_rgb(STATE_COLORS[state]) for state in range(len(states))])
    rgb = np.rint(np.einsum("syx,sc->yxc", weights, colors))
    opacity = np.round((total / max(total.max(), 1)) ** OPACITY_EXPONENT, 3)
    width = edges[1] - edges[0]
    hover = np.array(
        [
            "<br>".join(f"state {state}: {int(n)}" for state, n in enumerate(cell))
            for cell in counts.reshape(len(states), -1).T
        ]
    ).reshape(total.shape)
    return go.Image(
        z=np.concatenate([rgb, opacity[..., np.newaxis]], axis=-1),
        colormodel="rgba",
        x0=edges[0, 0] + width[0] / 2,
        dx=width[0],
        y0=edges[0, 1] + width[1] / 2,
        dy=width[1],
        hovertext=hover,
        hovertemplate="i: %{x:.3g}<br>q: %{y:.3g}<br>%{hovertext}<extra></extra>",
        name="shots",
    )


def _boundary_line(grid: npt.NDArray, predictions: npt.NDArray) -> npt.NDArray | None:
    """Point and direction of a linear decision boundary.

    The boundary is estimated from the classifier predictions on a square grid, as the
    line best fitting the midpoints between neighboring grid points predicted in
    different states. Returns ``None`` if the grid contains a single state.
    """
    size = int(np.sqrt(len(grid)))
    points = grid.reshape(size, size, 2)
    states = np.asarray(predictions).reshape(size, size)
    midpoints = []
    for axis in (0, 1):
        first = [slice(None)] * 2
        second = [slice(None)] * 2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        changed = states[tuple(first)] != states[tuple(second)]
        midpoints.append(
            (points[tuple(first)][changed] + points[tuple(second)][changed]) / 2
        )
    boundary = np.concatenate(midpoints)
    if len(boundary) < 2:
        return None
    center = boundary.mean(axis=0)
    # principal direction of the midpoints, i.e. the total least squares line
    direction = np.linalg.svd(boundary - center)[2][0]
    return np.array([center, direction])


def _clip(line: npt.NDArray, low: npt.NDArray, high: npt.NDArray) -> npt.NDArray | None:
    """Segment of a line within a rectangle, or ``None`` if outside."""
    center, direction = line
    with np.errstate(divide="ignore", invalid="ignore"):
        bounds = np.sort(
            np.stack([(low - center) / direction, (high - center) / direction]), axis=0
        )
    finite = np.isfinite(bounds).all(axis=0)
    start, stop = bounds[0][finite].max(), bounds[1][finite].min()
    if start >= stop:
        return None
    return center + np.outer([start, stop], direction)


def _boundaries(
    data: Data, qubit: QubitId, fit: Results, low: npt.NDArray, high: npt.NDArray
) -> list[go.Scatter]:
    """Decision boundaries of the trained classifiers, within the plot range.

    The available classifiers are linear, so their boundary is a straight line.
    Filled decision regions were only needed for non-linear classifiers, possibly
    predicting disconnected regions.
    """
    grid = evaluate_grid(data.data[qubit])
    traces = []
    for i, (model, predictions) in enumerate(
        zip(data.classifiers_list, fit.grid_preds[qubit])
    ):
        line = _boundary_line(grid, predictions)
        segment = _clip(line, low, high) if line is not None else None
        if segment is None:
            continue
        traces.append(
            go.Scatter(
                x=segment[:, 0],
                y=segment[:, 1],
                mode="lines",
                line={
                    "width": 2,
                    "dash": BOUNDARY_DASHES[i % len(BOUNDARY_DASHES)],
                    "color": BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)],
                },
                name=f"{run.pretty_name(model)} boundary",
                hoverinfo="skip",
            )
        )
    return traces


def _iq_plane(
    data: Data, qubit: QubitId, states: list[npt.NDArray], fit: Results | None
) -> go.Figure:
    """Shots densities in the IQ plane, with centroids and decision boundaries."""
    fig = go.Figure()
    edges = _edges(states)
    low, high = edges[0], edges[-1]
    pad = PADDING * (high - low)
    low, high = low - pad, high + pad
    fig.add_trace(_density(states, edges))
    for state, points in enumerate(states):
        name = f"state {state}"
        color = STATE_COLORS[state]
        centroid = np.mean(points, axis=0)
        fig.add_trace(
            go.Scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                name=name,
                legendgroup=name,
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": 12,
                    "color": color,
                    "line": {"width": 1, "color": "white"},
                },
            )
        )
    if fit is not None:
        # on top of the densities and the centroids
        for boundary in _boundaries(data, qubit, fit, low, high):
            fig.add_trace(boundary)

    # same units on both axes, since each state distribution is expected to be
    # approximately round: the plot ranges are extended to fill the available width
    fig.update_xaxes(title_text="i [a.u.]", range=[low[0], high[0]], constrain="range")
    fig.update_yaxes(
        title_text="q [a.u.]",
        range=[low[1], high[1]],
        autorange=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="range",
    )
    fig.update_layout(
        height=HEIGHT,
        margin=MARGIN,
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "x": 0,
            "font": {"size": LEGEND_FONT_SIZE},
        },
    )
    return fig


def _histogram(qubit: QubitId, states: list[npt.NDArray], fit: Results) -> go.Figure:
    """Histogram of the shots projected on the axis connecting the two centroids.

    The projection is centered on the classification threshold.
    """
    angle, threshold = fit.rotation_angle[qubit], fit.threshold[qubit]
    c, s = np.cos(angle), np.sin(angle)
    projections = [points @ np.array([c, -s]) - threshold for points in states]
    edges = np.histogram_bin_edges(np.concatenate(projections), bins=HISTOGRAM_BINS)
    centers = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    for state, projection in enumerate(projections):
        name = f"state {state}"
        color = STATE_COLORS[state]
        hist, _ = np.histogram(projection, bins=edges, density=True)
        fig.add_trace(
            go.Bar(
                x=centers,
                y=hist,
                width=edges[1] - edges[0],
                name=name,
                legendgroup=name,
                marker={"color": _rgba(color, 0.5)},
            )
        )
        mu, std = scipy_norm.fit(projection)
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=scipy_norm.pdf(centers, mu, std),
                name=f"{name} fit",
                legendgroup=name,
                showlegend=False,
                mode="lines",
                line={"width": 2, "color": color},
            )
        )
    fig.add_vline(x=0, line={"color": "black", "width": 2, "dash": "dot"})
    fig.update_xaxes(title_text="projection on the states axis [a.u.]")
    fig.update_yaxes(title_text="density")
    fig.update_layout(
        height=HISTOGRAM_HEIGHT,
        margin=MARGIN,
        barmode="overlay",
        bargap=0,
        legend={"font": {"size": LEGEND_FONT_SIZE}},
    )
    return fig


def _benchmarks(qubit: QubitId, models: list[str], fit: Results) -> go.Figure:
    """Accuracy and timings of the classifiers."""
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=SPACING,
        subplot_titles=("accuracy", "testing time [s]", "training time [s]"),
    )
    for i, model in enumerate(models):
        for plot in range(3):
            fig.add_trace(
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
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=3)
    fig.update_layout(margin=MARGIN)
    return fig


def plot_results(
    data: Data, qubit: QubitId, qubit_states: int, fit: Results | None
) -> list[go.Figure]:
    """Plots for the qubit and qutrit classification.

    Args:
        data (Data): acquisition data
        qubit (QubitID): qubit
        qubit_states (int): number of prepared states
        fit (Results): fit results
    """
    qubit_data = data.data[qubit]
    states = [
        np.column_stack([points["i"], points["q"]])
        for points in (
            qubit_data[qubit_data["state"] == s] for s in range(qubit_states)
        )
    ]

    figures = [_iq_plane(data, qubit, states, fit)]
    if fit is not None and qubit in getattr(fit, "rotation_angle", {}):
        figures.append(_histogram(qubit, states, fit))
    if fit is not None and len(data.classifiers_list) != 1:
        figures.append(_benchmarks(qubit, data.classifiers_list, fit))
    return figures
