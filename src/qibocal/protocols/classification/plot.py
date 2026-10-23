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
    TITLE_SIZE,
    evaluate_grid,
    get_color_state1,
)

__all__ = ["plot_results"]

STATE_COLORS = px.colors.qualitative.Plotly
"""Colors of the prepared states."""
BOUNDARY_COLORS = px.colors.qualitative.Dark2
"""Colors of the classifiers' decision boundaries."""
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


def _density(
    points: npt.NDArray, edges: npt.NDArray, color: str, name: str
) -> go.Heatmap:
    """Density of the shots of a single prepared state.

    The color goes from transparent to the state color, on a logarithmic scale, such
    that the tails of the distribution remain visible, and the densities of different
    states blend where overlapping. Only the bin counts are stored in the figure, so
    its size does not depend on the number of shots.
    """
    counts, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=[edges[:, 0], edges[:, 1]]
    )
    counts = counts.T
    with np.errstate(divide="ignore"):
        z = np.where(counts > 0, np.log10(counts), np.nan)
    centers = (edges[:-1] + edges[1:]) / 2
    return go.Heatmap(
        x=centers[:, 0],
        y=centers[:, 1],
        z=z,
        text=counts.astype(int),
        zmin=0,
        colorscale=[[0, _rgba(color, 0.15)], [1, _rgba(color, 0.85)]],
        showscale=False,
        name=name,
        legendgroup=name,
        hovertemplate="i: %{x:.3g}<br>q: %{y:.3g}<br>shots: %{text}<extra>"
        + name
        + "</extra>",
    )


def _boundaries(data: Data, qubit: QubitId, fit: Results) -> list[go.Contour]:
    """Decision boundaries of the trained classifiers.

    The available classifiers are linear, so their boundary is a line, drawn as the
    contour separating the two predicted states. Filled decision regions were only
    needed for non-linear classifiers, possibly predicting disconnected regions.
    """
    grid = evaluate_grid(data.data[qubit])
    return [
        go.Contour(
            x=grid[:, 0],
            y=grid[:, 1],
            z=np.array(predictions).flatten(),
            contours={"start": 0.5, "end": 0.5, "size": 1, "coloring": "lines"},
            line={"width": 2, "dash": "dash", "color": BOUNDARY_COLORS[i]},
            colorscale=[[0, BOUNDARY_COLORS[i]], [1, BOUNDARY_COLORS[i]]],
            showscale=False,
            name=f"{run.pretty_name(model)} boundary",
            showlegend=True,
            hoverinfo="skip",
        )
        for i, (model, predictions) in enumerate(
            zip(data.classifiers_list, fit.grid_preds[qubit])
        )
    ]


def _iq_plane(
    data: Data, qubit: QubitId, states: list[npt.NDArray], fit: Results | None
) -> go.Figure:
    """Shots densities in the IQ plane, with centroids and decision boundaries."""
    fig = go.Figure()
    edges = _edges(states)
    for state, points in enumerate(states):
        name = f"state {state}"
        color = STATE_COLORS[state]
        fig.add_trace(_density(points, edges, color, name))
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
        for boundary in _boundaries(data, qubit, fit):
            fig.add_trace(boundary)

    # same units on both axes, since each state distribution is expected to be
    # approximately round: the plot ranges are extended to fill the available width
    low, high = edges[0], edges[-1]
    pad = PADDING * (high - low)
    low, high = low - pad, high + pad
    fig.update_xaxes(title_text="i [a.u.]", range=[low[0], high[0]], constrain="range")
    fig.update_yaxes(
        title_text="q [a.u.]",
        range=[low[1], high[1]],
        scaleanchor="x",
        scaleratio=1,
        constrain="range",
    )
    fig.update_layout(
        height=HEIGHT,
        title={"text": "IQ plane", "font": {"size": TITLE_SIZE}},
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
        barmode="overlay",
        bargap=0,
        title={"text": "Projected histogram", "font": {"size": TITLE_SIZE}},
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
    fig.update_layout(title={"text": "Benchmarks", "font": {"size": TITLE_SIZE}})
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
