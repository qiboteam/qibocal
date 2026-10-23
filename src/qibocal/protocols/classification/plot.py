"""Plots for single-shot classification."""

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from scipy.stats import median_abs_deviation
from scipy.stats import norm as scipy_norm

from qibocal.auto.operation import Data, QubitId, Results

from ..utils import LEGEND_FONT_SIZE, project

__all__ = ["plot_results"]

STATE_COLORS = px.colors.qualitative.Plotly
"""Colors of the prepared states."""
BOUNDARY_COLOR = px.colors.qualitative.Dark2[0]
"""Color of the decision boundary."""
MARGIN = {"t": 30}
"""Figures' margins, with no title."""
HEIGHT = 600
"""Height of the IQ plane figure, in pixels."""
HISTOGRAM_HEIGHT = 400
"""Height of the projected histogram figure, in pixels."""
BINS_PER_STD = 3
"""Largest number of bins per standard deviation of the narrowest state distribution.

With many shots, bins are not made narrower than this, since a finer resolution is
not needed to resolve the states.
"""
MIN_BINS = 10
MAX_BINS = 200
"""Bounds on the number of bins per axis, both for the density and the histogram."""
PADDING = 0.05
"""Fraction of the data extent left empty around the IQ plane plot."""
OPACITY_EXPONENT = 0.6
"""Exponent of the opacity in the fraction of the maximum count, cf. :func:`_density`.

A linear opacity (``1``) hides the tails of the distributions, while a logarithmic
one over-represents them, merging the cores of overlapping states.
"""


def _rgba(color: str, alpha: float) -> str:
    """Convert a hex color to an RGBA string."""
    red, green, blue = hex_to_rgb(color)
    return f"rgba({red},{green},{blue},{alpha})"


def _bin_width(spread: float, shots: int, dimensions: int) -> float:
    """Bin width adapted to the number of shots, following Scott's rule.

    The optimal width of a histogram of normally distributed samples is
    ``3.5 sigma / n ** (1 / (d + 2))``, in ``d`` dimensions. Bins are not made narrower
    than :data:`BINS_PER_STD` per standard deviation.
    """
    scott = 3.5 * spread / max(shots, 1) ** (1 / (dimensions + 2))
    return max(scott, spread / BINS_PER_STD)


def _narrowest_spread(states: list[npt.NDArray], extent: float) -> float:
    """Smallest standard deviation of the states, in any coordinate.

    It is estimated from the median absolute deviation, insensitive to a minority of
    outliers.
    """
    spreads = [
        float(s)
        for state in states
        for s in np.atleast_1d(median_abs_deviation(state, axis=0, scale="normal"))
    ]
    return min((s for s in spreads if s > 0), default=extent)


def _edges(states: list[npt.NDArray]) -> npt.NDArray:
    """Square density bins, shared by all the states and both axes.

    The bin width is related to the narrowest state distribution, such that each blob
    is resolved, independently of how far apart the blobs are, and it grows with fewer
    shots, such that bins are not too sparse, cf. :func:`_bin_width`.
    """
    points = np.concatenate(states)
    low, high = points.min(axis=0), points.max(axis=0)
    extent = float(np.max(high - low)) or 1.0
    shots = min(len(state) for state in states)
    width = _bin_width(_narrowest_spread(states, extent), shots, dimensions=2)
    bins = int(np.clip(np.ceil(extent / width), MIN_BINS, MAX_BINS))
    width = extent / bins
    return low + width * np.arange(bins + 1)[:, np.newaxis]


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
    colors = np.array(
        [hex_to_rgb(STATE_COLORS[state]) for state in range(len(states))], dtype=float
    )
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


def _boundary(
    qubit: QubitId, fit: Results, low: npt.NDArray, high: npt.NDArray
) -> go.Scatter | None:
    """Decision boundary of the linear classifier, within the plot range.

    The boundary is the straight line where the shots' projection on the rotated I
    axis equals the threshold. Filled decision regions were only needed for
    non-linear classifiers, possibly predicting disconnected regions.
    """
    angle, threshold = fit.rotation_angle[qubit], fit.threshold[qubit]
    normal = np.array([np.cos(angle), -np.sin(angle)])
    line = np.array([threshold * normal, [normal[1], -normal[0]]])
    segment = _clip(line, low, high)
    if segment is None:
        return None
    return go.Scatter(
        x=segment[:, 0],
        y=segment[:, 1],
        mode="lines",
        line={"width": 2, "dash": "dash", "color": BOUNDARY_COLOR},
        name="boundary",
        hoverinfo="skip",
    )


def _centers(
    qubit: QubitId, states: list[npt.NDArray], fit: Results | None
) -> tuple[list[npt.NDArray], str]:
    """Centers of the prepared states, and their description.

    With a consistent Gaussian mixture fit, the centers lie at the means of the fitted
    Gaussians, along the states axis, and at the median of each preparation across
    it, such that spurious populations do not displace them. Otherwise, they are the
    plain averages of each preparation.
    """
    mixture = (
        fit.gaussian_mixture(qubit, consistent=True)
        if fit is not None and hasattr(fit, "gaussian_mixture")
        else None
    )
    if mixture is None:
        return [np.mean(points, axis=0) for points in states], "average"
    c, s = np.cos(fit.rotation_angle[qubit]), np.sin(fit.rotation_angle[qubit])
    centers = []
    for mean, points in zip(mixture.means, states):
        # coordinate across the states axis, then back to the IQ plane
        across = float(np.median(points @ np.array([s, c])))
        centers.append(np.array([c * mean + s * across, -s * mean + c * across]))
    return centers, "Gaussian mean"


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
    centers, center_kind = _centers(qubit, states, fit)
    for state, centroid in enumerate(centers):
        name = f"state {state}"
        color = STATE_COLORS[state]
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
                hovertemplate=f"{center_kind}<br>i: %{{x:.3g}}<br>q: %{{y:.3g}}"
                f"<extra>{name}</extra>",
            )
        )
    if fit is not None:
        # on top of the densities and the centroids
        boundary = _boundary(qubit, fit, low, high)
        if boundary is not None:
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

    The projection is centered on the threshold of the naive linear classifier. With a
    Gaussian mixture fit, its densities are shown as well, together with its threshold
    if the fit is consistent with the data. Otherwise, each state is fitted with a
    single Gaussian.
    """
    threshold = fit.threshold[qubit]
    projections = [
        project(points, fit.rotation_angle[qubit]) - threshold for points in states
    ]
    mixture = fit.gaussian_mixture(qubit) if hasattr(fit, "gaussian_mixture") else None
    points = np.concatenate(projections)
    extent = float(np.ptp(points)) or 1.0
    shots = min(len(projection) for projection in projections)
    spread = _narrowest_spread([p[:, np.newaxis] for p in projections], extent)
    bins = int(
        np.clip(
            np.ceil(extent / _bin_width(spread, shots, dimensions=1)),
            MIN_BINS,
            MAX_BINS,
        )
    )
    edges = np.histogram_bin_edges(points, bins=bins)
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
        if mixture is not None:
            density = mixture.pdf(centers + threshold, state)
        else:
            density = scipy_norm.pdf(centers, *scipy_norm.fit(projection))
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=density,
                name=f"{name} fit",
                legendgroup=name,
                showlegend=False,
                mode="lines",
                line={"width": 2, "color": color},
            )
        )
    fig.add_vline(
        x=0,
        line={"color": "black", "width": 2, "dash": "dot"},
        annotation_text="naive" if mixture is not None else None,
    )
    if mixture is not None and mixture.consistent:
        fig.add_vline(
            x=mixture.threshold - threshold,
            line={"color": "black", "width": 2},
            annotation_text="Gaussian",
            annotation_position="top left",
        )
    fig.update_xaxes(title_text="projection on the states axis [a.u.]")
    fig.update_yaxes(title_text="density")
    fig.update_layout(
        height=HISTOGRAM_HEIGHT,
        margin=MARGIN,
        plot_bgcolor="white",
        barmode="overlay",
        bargap=0,
        legend={"font": {"size": LEGEND_FONT_SIZE}},
    )
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
    if fit is not None:
        figures.append(_histogram(qubit, states, fit))
    return figures
