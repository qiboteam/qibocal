from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import scipy.constants as const
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseId,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.stats import theilslopes
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)

__all__ = ["coupler_anticrossing"]


CouplerSpectroscopyType = np.dtype(
    [
        ("bias", np.float64),
        ("frequency", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)


@dataclass
class CouplerAnticrossingParameters(Parameters):
    """Coupler anticrossing runcard inputs."""

    pulse_amplitude: float
    pulse_duration: float
    frequency: RangeLike
    bias: RangeLike

    def bias_span(self, center: float | None = None) -> Range:
        return to_range(self.bias, center)

    def frequency_span(self, center: float | None = None) -> Range:
        return to_range(self.frequency, center)


@dataclass
class CouplerAnticrossingResults(Results):
    """Crossings and the three asymptotes; frequencies in Hz, biases in a.u."""

    crossing_biases: dict[QubitPairId, list[float]] = field(default_factory=dict)
    crossing_frequencies: dict[QubitPairId, list[float]] = field(default_factory=dict)
    crossing_bias_errors: dict[QubitPairId, list[float]] = field(default_factory=dict)
    peak_biases: dict[QubitPairId, list[float]] = field(default_factory=dict)
    peak_frequencies: dict[QubitPairId, list[float]] = field(default_factory=dict)
    cluster_labels: dict[QubitPairId, list[int]] = field(default_factory=dict)
    asymptotes: dict[QubitPairId, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )

    def __contains__(self, key):
        """Keep crossing availability independent of the cluster diagnostics."""
        return key in self.crossing_biases


@dataclass
class CouplerAnticrossingData(Data):
    """Coupler anticrossing acquisition outputs."""

    data: dict[QubitPairId, np.ndarray] = field(default_factory=dict)


def _signal_grid(record: np.ndarray):
    """Mitigate bias-independent background and return PCA components on-grid."""

    bias = np.unique(record["bias"])
    frequency = np.unique(record["frequency"])
    dims = (len(bias), len(frequency))
    coordinates = np.column_stack((record["bias"], record["frequency"]))
    if (
        min(dims) < 9
        or len(record) != dims[0] * dims[1]
        or len(np.unique(coordinates, axis=0)) != len(record)
        or not all(np.all(np.isfinite(record[name])) for name in record.dtype.names)
    ):
        raise ValueError("A finite, complete bias-frequency grid is required")
    order = np.lexsort((record["frequency"], record["bias"]))
    quadratures = np.stack((record["i"][order], record["q"][order]), axis=-1).reshape(
        dims + (2,)
    )

    # The median across bias sweeps estimates the common frequency-dependent background.
    background = np.median(quadratures, axis=1, keepdims=True)
    quadratures = quadratures - background

    pca = PCA(n_components=2)
    components = pca.fit_transform(quadratures.reshape((-1, 2)))
    components = components.reshape(dims + (2,))

    return bias, frequency, components


def _extract_peaks(biases, frequencies, signal):
    """Keep significant extrema supported by peaks in nearby bias columns."""
    smooth = gaussian_filter1d(signal, 1, axis=1)
    differences = np.diff(signal, axis=1).ravel()
    noise = (
        1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
    )
    threshold = max(3 * noise, np.finfo(float).eps * np.max(np.abs(signal)) * 10)
    peak_columns, peak_bins, peak_prominences = [], [], []
    for column, row in enumerate(smooth):
        positive, positive_properties = find_peaks(row, prominence=threshold)
        negative, negative_properties = find_peaks(-row, prominence=threshold)
        peaks = np.concatenate((positive, negative))
        prominences = np.concatenate(
            (positive_properties["prominences"], negative_properties["prominences"])
        )
        peak_columns.extend([column] * len(peaks))
        peak_bins.extend(peaks)
        peak_prominences.extend(prominences)
    columns = np.asarray(peak_columns, dtype=int)
    bins = np.asarray(peak_bins, dtype=int)
    prominences = np.asarray(peak_prominences)

    if not len(columns):
        return np.array([]), np.array([]), prominences

    # Use sweep resolution, not raw Hz/bias units, to define local support.
    # Scaling by 2 bias steps and 3 frequency bins makes the later radius-1
    # Chebyshev query span those respective tolerances; the larger frequency
    # window accommodates peak-bin jitter between adjacent bias columns.
    points = np.column_stack(
        (
            (biases[columns] - biases[0]) / (2 * np.median(np.diff(biases))),
            (frequencies[bins] - frequencies[0])
            / (3 * np.median(np.diff(frequencies))),
        )
    )
    neighbors = cKDTree(points).query_ball_point(points, r=1 + 1e-9, p=np.inf)
    supported = np.array(
        [len(np.unique(columns[indices])) >= 3 for indices in neighbors]
    )
    return (
        biases[columns[supported]],
        frequencies[bins[supported]],
        prominences[supported],
    )


def _cluster_peaks(biases, frequencies, peak_biases, peak_frequencies):
    """Select three substantial spatial clusters, ordered from left to right."""
    if len(peak_biases) < 18:
        raise ValueError("Too few peaks for three independent clusters")
    bias_step = np.median(np.diff(biases))
    frequency_step = np.median(np.diff(frequencies))
    points = np.column_stack(
        (
            (peak_biases - biases[0]) / (2 * bias_step),
            (peak_frequencies - frequencies[0]) / (3 * frequency_step),
        )
    )
    labels = DBSCAN(eps=1 + 1e-9, min_samples=3, metric="chebyshev").fit_predict(points)
    candidates = [
        label
        for label in np.unique(labels)
        if label >= 0 and len(np.unique(peak_biases[labels == label])) >= 6
    ]
    centers = {label: np.median(peak_biases[labels == label]) for label in candidates}
    spans = {label: np.ptp(peak_biases[labels == label]) for label in candidates}
    best = None
    best_size = 0
    for group in combinations(candidates, 3):
        if any(
            abs(centers[left] - centers[right])
            <= max(2 * bias_step, (spans[left] + spans[right]) / 4)
            for left, right in combinations(group, 2)
        ):
            continue
        size = sum(np.sum(labels == label) for label in group)
        if size > best_size:
            best, best_size = group, size
    if best is None:
        raise ValueError(
            "Three distinct, sufficiently supported clusters are not resolved"
        )
    selected = np.full(len(labels), -1, dtype=int)
    for index, label in enumerate(sorted(best, key=centers.__getitem__)):
        selected[labels == label] = index
    return selected


def _hyperbola(x, y0, m, xc, g, k, branch):
    """Tilted hyperbola: y0 + m*(x-xc) +/- sqrt(g**2 + k**2*(x-xc)**2)."""
    displacement = x - xc
    return y0 + m * displacement + branch * np.hypot(g, k * displacement)


def _fit_asymptotes(peak_biases, peak_frequencies, labels):
    """Robust straight line (frequency = intercept + slope * bias) per cluster."""
    asymptotes = {}
    for cluster in range(3):
        mask = labels == cluster
        slope, intercept, low, high = theilslopes(
            peak_frequencies[mask], peak_biases[mask]
        )
        asymptotes[str(cluster)] = {
            "slope": float(slope),
            "slope_error": float((high - low) / 2),
            "intercept": float(intercept),
        }
    return asymptotes


def _asymptote_crossings(biases, asymptotes):
    """Crossings are the intersections of neighbouring asymptotes (0-1 and 1-2).

    They may lie outside the scanned range when the asymptotes are poorly constrained.
    """
    crossing_biases, crossing_frequencies = [], []
    for left, right in (("0", "1"), ("1", "2")):
        first, second = asymptotes[left], asymptotes[right]
        if first["slope"] == second["slope"]:
            raise ValueError("Neighbouring asymptotes are parallel")
        bias = (second["intercept"] - first["intercept"]) / (
            first["slope"] - second["slope"]
        )
        crossing_biases.append(float(bias))
        crossing_frequencies.append(float(first["intercept"] + first["slope"] * bias))
    errors = [float(np.median(np.diff(biases))) / 2] * 2
    return crossing_biases, crossing_frequencies, errors


def _acquisition(
    params: CouplerAnticrossingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerAnticrossingData:
    """Protocol to perform sweep to locate coupler."""

    coupler_pulse = Pulse(
        duration=params.pulse_duration,
        amplitude=params.bias_span()[0],
        envelope=Rectangular(),
    )

    full_sequence = PulseSequence()
    freq_parsweepers: dict[QubitPairId, Sweeper] = {}
    ro_pulses_id: dict[QubitPairId, PulseId] = {}
    updates: dict = {}
    for pair in targets:
        qubit, _ = pair
        natives = platform.natives.single_qubit[qubit]
        qd_channel, _ = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        coupler_id = platform.calibration.two_qubits[pair].coupler
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        coupler_ch = platform.couplers[coupler_id].flux
        assert coupler_ch is not None, "Coupler channel should not be None"

        updates |= {
            platform.qubits[qubit].probe: {
                "frequency": readout_frequency(qubit, platform)
            }
        }

        spectr_pulse = Pulse(
            duration=params.pulse_duration,
            amplitude=params.pulse_amplitude,
            envelope=Rectangular(),
        )

        full_sequence += PulseSequence(
            [
                (qd_channel, spectr_pulse),
                (coupler_ch, coupler_pulse),
                (ro_channel, Delay(duration=spectr_pulse.duration)),
                (ro_channel, ro_pulse),
            ]
        )
        ro_pulses_id[pair] = ro_pulse.id

        qubit_freq = platform.config(qd_channel).frequency
        # define the parameters to sweep and their range:
        freq_parsweepers[pair] = Sweeper(
            parameter=Parameter.frequency,
            range=params.frequency_span(qubit_freq),
            channels=[qd_channel],
        )

    bias_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.bias_span(),
        pulses=[coupler_pulse],
    )

    data = CouplerAnticrossingData()
    results = platform.execute(
        [full_sequence],
        [[bias_sweeper], list(freq_parsweepers.values())],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    bias_values = np.array(bias_sweeper.values)
    for pair in targets:
        freq_values = np.array(freq_parsweepers[pair].values)

        quadratures = np.asarray(results[ro_pulses_id[pair]])

        # Store absolute bias here; fitting and updates must not add DC again.
        data.register_qubit(
            CouplerSpectroscopyType,
            pair,
            {
                "frequency": np.tile(freq_values, len(bias_values)),
                "bias": np.repeat(bias_values, len(freq_values)),
                "i": quadratures[..., 0].ravel(),
                "q": quadratures[..., 1].ravel(),
            },
        )

    return data


def _fit(data: CouplerAnticrossingData) -> CouplerAnticrossingResults:
    """Identify the three asymptotes and the two qubit-coupler avoided crossings."""

    results = CouplerAnticrossingResults()
    for pair, records in data.data.items():
        try:
            biases, frequencies, components = _signal_grid(records)
            peak_biases, peak_frequencies, _ = _extract_peaks(
                biases, frequencies, components[..., 0]
            )
            results.peak_biases[pair] = peak_biases.tolist()
            results.peak_frequencies[pair] = peak_frequencies.tolist()
            labels = _cluster_peaks(biases, frequencies, peak_biases, peak_frequencies)
            asymptotes = _fit_asymptotes(peak_biases, peak_frequencies, labels)
            results.cluster_labels[pair] = labels.tolist()
            results.asymptotes[pair] = asymptotes
            crossings = _asymptote_crossings(biases, asymptotes)
        except ValueError as error:
            log.warning(f"Anticrossing fit failed for pair {pair}: {error}")
            continue
        (
            results.crossing_biases[pair],
            results.crossing_frequencies[pair],
            results.crossing_bias_errors[pair],
        ) = crossings
    return results


CLUSTER_COLORS = ["#00CC96", "#FFA15A", "#AB63FA"]


def _add_peaks(figure, fit, target, bias_range):
    """Overlay extracted peaks, coloured per cluster, with their asymptotes."""
    peak_biases = np.asarray(fit.peak_biases[target])
    peak_frequencies = np.asarray(fit.peak_frequencies[target]) * const.nano
    figure.add_scatter(
        x=peak_biases,
        y=peak_frequencies,
        mode="markers",
        marker={"size": 4, "color": "white", "line": {"width": 0.5}},
        name="Extracted signal",
    )
    if target not in fit.cluster_labels:
        return
    labels = np.asarray(fit.cluster_labels[target])
    for cluster, color in enumerate(CLUSTER_COLORS):
        mask = labels == cluster
        group = f"cluster-{cluster}"
        figure.add_scatter(
            x=peak_biases[mask],
            y=peak_frequencies[mask],
            mode="markers",
            marker={"size": 5, "color": color},
            name=f"Cluster {cluster + 1}",
            legendgroup=group,
        )
        line = fit.asymptotes.get(target, {}).get(str(cluster))
        if line is not None:
            figure.add_scatter(
                x=bias_range,
                y=(line["intercept"] + line["slope"] * np.asarray(bias_range))
                * const.nano,
                mode="lines",
                line={"color": color, "width": 3},
                name=f"Asymptote {cluster + 1}",
                legendgroup=group,
            )


def _add_crossings(figure, fit, target):
    for index, (bias, frequency, error) in enumerate(
        zip(
            fit.crossing_biases[target],
            fit.crossing_frequencies[target],
            fit.crossing_bias_errors[target],
        ),
        1,
    ):
        figure.add_vline(
            x=bias, line_dash="dash", annotation_text=f"Crossing {index}: {bias:.4f}"
        )
        figure.add_scatter(
            x=[bias],
            y=[frequency * const.nano],
            error_x={"array": [error]},
            mode="markers",
            marker={"symbol": "x", "size": 10},
            name=f"Avoided crossing {index}",
        )


def _report(fit, target):
    report = ""
    if target in fit:
        report += table_html(
            table_dict(
                str(target),
                [
                    "Left crossing bias [a.u.]",
                    "Right crossing bias [a.u.]",
                    "Left crossing frequency [GHz]",
                    "Right crossing frequency [GHz]",
                    "Left bias half-bin resolution [a.u.]",
                    "Right bias half-bin resolution [a.u.]",
                ],
                [
                    *fit.crossing_biases[target],
                    *(np.asarray(fit.crossing_frequencies[target]) * const.nano),
                    *fit.crossing_bias_errors[target],
                ],
            )
        )
        report += (
            "<p>Crossings are the intersections of neighbouring asymptotes; "
            "half-bin resolution is not a statistical uncertainty.</p>"
        )
    for cluster, line in fit.asymptotes.get(target, {}).items():
        report += table_html(
            table_dict(
                f"{target}, asymptote {int(cluster) + 1}",
                [
                    "Slope [MHz/a.u.]",
                    "Slope 95% half-interval [MHz/a.u.]",
                    "Frequency at zero bias [GHz]",
                ],
                [
                    line["slope"] * 1e-6,
                    line["slope_error"] * 1e-6,
                    line["intercept"] * const.nano,
                ],
            )
        )
    return report


def _plot(
    data: CouplerAnticrossingData,
    target: QubitPairId,
    fit: CouplerAnticrossingResults | None = None,
):
    """Plotting function for CouplerAnticrossing Experiment."""
    try:
        biases, frequencies, components = _signal_grid(data.data[target])
    except ValueError as error:
        log.warning(f"Cannot plot anticrossing grid for pair {target}: {error}")
        return [], ""

    figure = go.Figure(
        go.Heatmap(
            x=biases,
            y=frequencies * const.nano,
            z=components[..., 0].T,
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "First PCA component [a.u.]"},
        )
    )
    figure.update_layout(
        title="Spectroscopy: first PCA component",
        xaxis_title="Coupler Bias [a.u.]",
        yaxis_title="Qubit Frequency [GHz]",
        xaxis_range=[biases[0], biases[-1]],
        yaxis_range=[frequencies[0] * const.nano, frequencies[-1] * const.nano],
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.22,
            "yanchor": "top",
        },
        margin={"b": 120},
    )
    report = ""
    if fit is not None:
        if target in fit.peak_biases:
            _add_peaks(figure, fit, target, [biases[0], biases[-1]])
        if target in fit:
            _add_crossings(figure, fit, target)
        report = _report(fit, target)
    return [figure], report


coupler_anticrossing = Protocol(_acquisition, _fit, _plot)
"""CouplerAnticrossing Routine object."""
