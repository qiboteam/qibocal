import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from scipy.optimize import OptimizeWarning, curve_fit

from ...auto.operation import Parameters
from ..utils import (
    DISTANCE_XY,
    DISTANCE_Z,
    HZ_TO_GHZ,
    FeatExtractionError,
    clustering,
    merging,
    minmax_scaling,
    peaks_finder,
    reshaping_raw_signal,
    zca_whiten,
)


@dataclass(kw_only=True)
class FluxFrequencySweepParameters(Parameters):
    """Parameters to define flux DC sweep."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [a.u.]."""
    bias_step: float
    """Bias step for sweep [a.u.]."""

    @property
    def frequency_range(self) -> np.ndarray:
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)

    @property
    def bias_range(self) -> np.ndarray:
        return np.arange(-self.bias_width / 2, self.bias_width / 2, self.bias_step)


def create_data_array(freq, bias, signal, dtype):
    """Create custom dtype array for acquired data."""
    size = len(freq) * len(bias)
    ar = np.empty(size, dtype=dtype)
    frequency, biases = np.meshgrid(freq, bias)
    ar["freq"] = frequency.ravel()
    ar["bias"] = biases.ravel()
    ar["signal"] = signal.ravel()
    return np.rec.array(ar)


def flux_dependence_plot(data, fit, qubit, fit_function=None):
    figures = []
    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=qubit_data.freq * HZ_TO_GHZ,
            y=qubit_data.bias,
            z=qubit_data.signal,
            colorbar={"title": "Signal [a.u.]"},
            colorscale="Viridis",
        ),
    )

    # TODO: This fit is for frequency, can it be reused here, do we even want the fit ?
    if (
        fit is not None
        and fit_function is not None
        and data.__class__.__name__ != "CouplerSpectroscopyData"
        and fit.successful_fit[qubit]
    ):
        params = fit.fitted_parameters[qubit]
        bias = np.unique(qubit_data.bias)
        fig.add_trace(
            go.Scatter(
                x=fit_function(bias, **params),
                y=bias,
                showlegend=True,
                name="Fit",
                marker={"color": "rgb(248, 248, 248)"},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    fit.frequency[qubit] * HZ_TO_GHZ,
                ],
                y=[
                    fit.sweetspot[qubit],
                ],
                mode="markers",
                marker={
                    "size": 8,
                    "color": "red",
                },
                name="Sweetspot",
                showlegend=True,
            ),
        )

    fig.update_xaxes(
        title_text="Frequency [GHz]",
    )
    fig.update_yaxes(title_text="Bias [a.u.]")

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})

    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h"},
    )

    figures.append(fig)

    return figures


def flux_crosstalk_plot(data, qubit, fit, fit_function):
    figures = []
    fitting_report = ""
    all_qubit_data = {
        index: data_qubit
        for index, data_qubit in data.data.items()
        if index[0] == qubit
    }
    fig = make_subplots(
        rows=1,
        cols=len(all_qubit_data),
        horizontal_spacing=0.3 / len(all_qubit_data),
        vertical_spacing=0.1,
        subplot_titles=len(all_qubit_data) * ("Signal [a.u.]",),
    )
    for col, (flux_qubit, qubit_data) in enumerate(all_qubit_data.items()):
        frequencies = qubit_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=qubit_data.bias,
                z=qubit_data.signal,
                showscale=False,
            ),
            row=1,
            col=col + 1,
        )
        if fit is not None and fit.successful_fit[qubit] and flux_qubit[1] != qubit:
            fig.add_trace(
                go.Scatter(
                    x=fit_function(
                        xj=qubit_data.bias, **fit.fitted_parameters[flux_qubit]
                    ),
                    y=qubit_data.bias,
                    showlegend=not any(
                        isinstance(trace, go.Scatter) for trace in fig.data
                    ),
                    legendgroup="Fit",
                    name="Fit",
                    marker={"color": "green"},
                ),
                row=1,
                col=col + 1,
            )

        fig.update_xaxes(
            title_text="Frequency [GHz]",
            row=1,
            col=col + 1,
        )

        fig.update_yaxes(
            title_text=f"Qubit {flux_qubit[1]}: Bias [a.u.]", row=1, col=col + 1
        )

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis2={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis3={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(
        showlegend=True,
    )
    figures.append(fig)

    return figures, fitting_report


def G_f_d(xi, xj, offset, d, crosstalk_element, normalization):
    """Auxiliary function to calculate qubit frequency as a function of bias.

    It also determines the flux dependence of :math:`E_J`,:math:`E_J(\\phi)=E_J(0)G_f_d`.
    For more details see: https://arxiv.org/pdf/cond-mat/0703002.pdf

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        offset (float): phase_offset [a.u.].
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        crosstalk_element(float): off-diagonal crosstalk matrix element
        normalization(float): diagonal crosstalk matrix element
    Returns:
        (float)
    """
    return (
        d**2
        + (1 - d**2)
        * np.cos(
            np.pi
            * (xi * normalization + normalization * xj * crosstalk_element + offset)
        )
        ** 2
    ) ** 0.25


def transmon_frequency(
    xi, xj, w_max, d, normalization, offset, crosstalk_element, charging_energy
):
    r"""Approximation to transmon frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        normalization(float): diagonal crosstalk matrix element
        offset (float): phase_offset [a.u.].
        crosstalk_element(float): off-diagonal crosstalk matrix element
        charging_energy (float): Ec / h (GHz)

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return (w_max + charging_energy) * G_f_d(
        xi,
        xj,
        offset=offset,
        d=d,
        normalization=normalization,
        crosstalk_element=crosstalk_element,
    ) - charging_energy


def transmon_readout_frequency(
    xi,
    xj,
    w_max,
    d,
    normalization,
    crosstalk_element,
    offset,
    resonator_freq,
    g,
    charging_energy,
):
    r"""Approximation to flux dependent resonator frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
         xi (float): bias of target qubit
         xj (float): bias of neighbor qubit
         w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
         d (float): asymmetry between the two junctions of the transmon.
                    Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
         normalization(float): diagonal crosstalk matrix element
         offset (float): phase_offset [a.u.].
         crosstalk_element(float): off-diagonal crosstalk matrix element
         resonator_freq (float): bare resonator frequency [GHz]
         g (float): readout coupling.
         charging_energy (float): Ec / h (GHz)

     Returns:
         (float): resonator frequency as a function of bias.
    """

    qubit_frequency = transmon_frequency(
        xi=xi,
        xj=xj,
        w_max=w_max,
        d=d,
        normalization=normalization,
        offset=offset,
        crosstalk_element=crosstalk_element,
        charging_energy=charging_energy,
    )
    return resonator_freq + g**2 * (
        1 / (resonator_freq - qubit_frequency)
        - 1 / (resonator_freq - qubit_frequency + charging_energy)
    )


def filter_data(matrix_z: np.ndarray):
    """Filter data with a ZCA transformation and then a unit-variance Gaussian."""

    # adding zca filter for filtering out background noise gradient
    zca_z = zca_whiten(matrix_z)
    # adding gaussian fliter with unitary variance for blurring the signal and reducing noise
    return ndimage.gaussian_filter(zca_z, 1)


def flux_extract_feature(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    find_min: bool,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features of the signal by filtering out background noise.

    It first applies a custom filter mask (see `custom_filter_mask`)
    and then finds the biggest peak for each DC bias value;
    the masked signal is then clustered (see `clustering`) in order to classify the relevant signal for the experiment.
    If `find_min` is set to `True` it finds minimum peaks of the input signal;
    `min_points` is the minimum number of points for a cluster to be considered relevant signal.
    Position of the relevant signal is returned.
    """

    reshaped_x, reshaped_y, reshaped_z = reshaping_raw_signal(x, y, z)
    reshaped_z = -reshaped_z if find_min else reshaped_z

    z_masked = filter_data(reshaped_z)

    # renormalizing
    z_masked_norm = minmax_scaling(z_masked, axis=1)

    # filter data using find_peaks
    peaks_dict = peaks_finder(reshaped_x, reshaped_y, z_masked_norm)
    if len(peaks_dict.keys()) == 0:  # if find_peaks fails
        """
        Peaks Detection Failed:
        no peaks found in peaks_finder routine.
        """
        return None, None

    peaks, labels = clustering(peaks_dict, z_masked)

    # merging close clusters
    try:
        signal_clusters = merging(
            peaks,
            labels,
            min_points,
            distance_xy=DISTANCE_XY,
            distance_z=DISTANCE_Z,
        )

    except FeatExtractionError:
        return None, None

    medians = np.array(
        [[lab, np.median(cl["cluster"][2, :])] for lab, cl in signal_clusters.items()]
    )

    signal_labels = np.zeros(labels.size, dtype=bool)
    signal_idx = medians[np.argmax(medians[:, 1]), 0]
    signal_labels[signal_clusters[signal_idx]["cluster"][-1, :].astype(int)] = True

    return peaks_dict["x"]["val"][signal_labels], peaks_dict["y"]["val"][signal_labels]


def _function_dof(fit_function) -> int:
    sig = inspect.signature(fit_function)

    # Filter for positional parameters without defaults
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]

    # Subtract 1 for the independent variable
    return len(params) - 1


def select_sweetspot(
    offset: float,
    normalization: float,
    bias_window: npt.ArrayLike,
    max_distance: float = 0,
):
    """Select the closest flux sweetspot that lies in the acquired bias window.

    The fitted model is periodic in the reduced flux, so every integer ``n`` gives
    a candidate ``(n - offset) / normalization``.

    If no sweetspot lies inside the acquired bias window, the closest sweetspot
    outside the window is selected unless it is farther than ``max_distance``.
    """
    low, high = np.sort(np.asarray(bias_window, dtype=float))

    n_low = int(np.ceil(offset + normalization * low))
    n_high = int(np.floor(offset + normalization * high))
    candidates = []
    for n in range(min(n_low, n_high), max(n_low, n_high) + 1):
        candidate = (n - offset) / normalization
        if low <= candidate <= high:
            candidates.append(candidate)

    if candidates:
        # If there is at least one sweetspot in the bias window, take the bias with abs
        # closest to 0.0
        return min(candidates, key=abs)

    # No sweetspot lies inside the acquired window. Find the closest one to the window.
    n = int(np.floor(offset + 0.5))
    sweetspot = (n - offset) / normalization

    # If the nearest sweetspot is outside the window, its distance to the acquired data
    # is measured from the closest window boundary.
    distance = max(low - sweetspot, 0, sweetspot - high)

    if distance > max_distance:
        raise ValueError("No fitted sweetspot lies inside the acquired bias window.")

    return sweetspot


def _continuity_score(
    xvals: npt.NDArray[np.floating],
    yvals: npt.NDArray[np.floating],
    inliers: npt.NDArray[np.bool_],
) -> int:
    """Score consecutive inlier runs quadratically, counting each y-value once."""
    order = np.argsort(xvals)  # in practice they are already ordered
    ordered_yvals = np.asarray(yvals)[order]
    ordered_inliers = np.asarray(inliers, dtype=bool)[order]

    # pad with 0s to ensure a change if the arc starts at the boundary
    padded = np.pad(ordered_inliers.astype(int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    unique_y_counts = [
        len(np.unique(ordered_yvals[start:stop])) for start, stop in zip(starts, stops)
    ]

    return int(np.sum(np.square(unique_y_counts)))


def ransac_fit(
    xvals: npt.NDArray[np.floating],
    yvals: npt.NDArray[np.floating],
    fit_function: Callable[..., np.ndarray],
    residual_threshold: float,
    min_trials: int = 100,
    max_trials: int = 5000,
    stop_probability: float = 0.999,
    random_state: int = 0,
    bounds: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
):
    """Fit a model to data using RANSAC, ignoring outliers.

    Repeatedly fits ``fit_function`` to minimal random subsets of the data (sized to the
    function's degrees of freedom). Points with residual below ``residual_threshold``
    are inliers. Candidate models are scored by summing the squared lengths of
    consecutive inlier runs along the x-axis, favoring a continuous feature over
    scattered noise.

    The number of trials adapts dynamically based on the current inlier
    ratio, following the standard RANSAC stopping criterion, and is bounded by
    ``min_trials`` and ``max_trials``. A final least-squares refit is performed on the
    best inlier set.

    Returns:
        Optimal fit parameters from the least-squares refit on the best inlier set.
    """
    # A fixed seed makes debugging and regression tests reproducible. RandomState's
    # output is guaranteed to be stable across numpy versions:
    # https://numpy.org/doc/2.5/reference/random/legacy.html#numpy.random.RandomState
    rng = np.random.RandomState(random_state)

    method = "lm" if bounds is None else "trf"
    fit_kwargs: dict[str, Any] = {} if bounds is None else {"bounds": bounds}

    function_dof = _function_dof(fit_function)

    # N_needed is the adaptively-updated trial budget; start at infinity so the loop is
    # initially bounded only by min_trials/max_trials.
    N_needed = np.inf
    ransac_iterations = 0
    best_inliers = np.zeros(len(xvals), dtype=bool)
    best_params = None
    best_score = (0, 0)
    # Track already-sampled subsets so we don't waste a trial refitting the exact same
    # minimal sample twice.
    tried_subsets = set()

    # Standard adaptive RANSAC loop: keep going until we've hit max_trials, or until the
    # estimated number of iterations needed to find an all-inlier sample (with
    # probability stop_probability) drops below where we already are, although we never
    # stop before attempting at least min_trials.
    # https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
    while (
        ransac_iterations < min(N_needed, max_trials) or ransac_iterations < min_trials
    ):
        ransac_iterations += 1

        # Draw a minimal sample because fewer samples means that the probability of all
        # points being on the feature of interest is maximized. To perform a fit we need
        # at least as many points as the model's dof.
        subset = rng.choice(len(xvals), function_dof, replace=False)
        subset_ = tuple(sorted(subset))
        if subset_ in tried_subsets:
            continue
        tried_subsets.add(subset_)

        try:
            with warnings.catch_warnings():
                # Poor fits are expected for random subsets, so suppress these warnings.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    fit_function,
                    xvals[subset],
                    yvals[subset],
                    method=method,
                    **fit_kwargs,
                )
        except RuntimeError:
            continue

        residuals_all = np.abs(yvals - fit_function(xvals, *popt))
        inliers = residuals_all < residual_threshold
        if inliers.sum() == len(xvals):
            # all points are inliers, so we cannot do better
            best_inliers = inliers
            best_params = popt
            break

        # Use a continuity score to reduce the sensitivity to a noisy background.
        # Especially if the arc is only in a small part of the bias range.
        # TODO: would be nice to have this as a callback
        unique_y_count = len(np.unique(yvals[inliers]))
        score = (
            _continuity_score(xvals, yvals, inliers),
            unique_y_count,
        )

        # Accept as a new best if it has a better continuity score and at least matches
        # the minimal sample size.
        if inliers.sum() >= len(subset) and score > best_score:
            best_inliers = inliers
            best_params = popt
            best_score = score
            # Re-estimate how many trials are needed to have `stop_probability`
            # confidence of drawing an all-inlier minimal sample.
            denom = np.log(1 - (best_inliers.sum() / len(xvals)) ** len(subset))
            N_needed = np.log(1 - stop_probability) / denom

    if best_params is None:
        raise RuntimeError(
            f"RANSAC failed to find a valid fit after {ransac_iterations} iterations: "
            f"no sample of size {function_dof} produced at least {function_dof} inliers "
            f"(residual_threshold={residual_threshold})."
        )

    # Finally optimize by doing a least-squares fit to the best set of inliers.
    popt, _ = curve_fit(
        fit_function,
        xvals[best_inliers],
        yvals[best_inliers],
        p0=best_params,
        method=method,
        maxfev=100000,
        **fit_kwargs,
    )

    return popt
