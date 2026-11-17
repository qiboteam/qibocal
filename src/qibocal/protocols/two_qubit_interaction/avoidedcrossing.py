"""Qubit-pair avoided crossings under a qubit flux pulse.

Experiment
    A long spectroscopy pulse on the lower-frequency qubit plays together with a
    flux pulse on the higher-frequency qubit, then both qubits are read out.
    The flux pulse amplitude (the *bias*, applied on top of the DC offset) and
    the spectroscopy drive frequency are swept.

Physics
    While the fluxed qubit is far detuned, the driven qubit line is flat. Where its
    frequency ``f_c(x)`` crosses the driven qubit frequency ``fq``, they hybridise with coupling
    ``g``; with ``dx = x - xc`` and ``sc = f_c'(xc)`` the bare fluxed-qubit slope, the two
    eigenbranches are

        f(x) = fq + sc*dx/2 + s*sqrt((sc*dx/2)**2 + g**2),   s = +1 or -1.

    The bright, qubit-like line follows ``s = -sign(sc*dx)``: it jumps by about
    ``-2 g sign(sc)`` at ``xc`` (jump sign opposite to ``sc``) and its tangent there is
    ``sc/2``. Hence each crossing gives a point ``f_c(xc) = fq`` of the fluxed-qubit curve
    and, when both branches are resolved, the slope of their sum gives the bare
    fluxed-qubit asymptote: ``d(f_+ + f_-)/dx = f_c'(x)``.

Analysis
    Extract peaks once from the mean of the sign-aligned readout PCA maps.
    Estimate crossings from minima of the branch separation or bright-branch
    jumps, and seed them with their median frequency and coupling. Split the
    extracted peaks halfway between adjacent crossings, the outer crossings also
    taking the extreme bias intervals, and RANSAC-fit each crossing's
    eigenbranches on its own peaks.

Units: frequencies in Hz, biases in a.u., slopes in Hz/a.u.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    GaussianSquare,
    Parameter,
    Pulse,
    PulseId,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.flux_dependence.utils import (
    adaptive_residual_threshold,
    filter_data,
    ransac_fit,
)
from qibocal.protocols.two_qubit_interaction.utils import order_pair
from qibocal.protocols.utils import (
    RangeLike,
    readout_frequency,
    table_html,
    to_range,
)

__all__ = ["avoided_crossing"]

AvoidedCrossingType = np.dtype(
    [
        ("bias", float),
        ("freq", float),
        ("low_i", float),
        ("low_q", float),
        ("high_i", float),
        ("high_q", float),
    ]
)
"""One map point: coupler bias, drive frequency and both readouts' I/Q components."""

WINDOW = 10
"""Maximum half-width of numerical asymptote estimates, in bias steps."""


@dataclass
class AvoidedCrossingParameters(Parameters):
    """Avoided-crossing spectroscopy inputs for a pair of qubits.

    The lower-frequency qubit is driven while the higher-frequency qubit's
    flux pulse amplitude is swept.
    """

    bias: RangeLike
    """Higher-frequency qubit flux pulse amplitude range [a.u.], applied on top
    of its DC flux offset."""
    frequency: RangeLike
    """Spectroscopy frequency range [Hz] for the lower-frequency qubit.

    Center-relative ranges use that qubit's configured drive frequency.
    """
    pulse_duration: float
    """Duration of both the spectroscopy drive and qubit flux pulses [ns]."""
    pulse_amplitude: float
    """Spectroscopy drive pulse amplitude on the lower-frequency qubit [a.u.]."""


@dataclass
class AvoidedCrossingData(Data):
    """Spectroscopy maps keyed by qubit pair ``(q0, q1)``."""

    data: dict[tuple, npt.NDArray] = field(default_factory=dict)

    @property
    def bias(self) -> npt.NDArray:
        """Return the bias values of the spectroscopy map."""
        return np.unique(next(iter(self.data.values()))["bias"])

    def pair_frequency(self, pair: QubitPairId) -> npt.NDArray:
        """Return the qubit frequency values of the spectroscopy map for the given pair."""
        return np.unique(self.data[pair]["freq"])

    def pair_signal(self, pair: QubitPairId) -> tuple[npt.NDArray, npt.NDArray]:
        """Return the I/Q signals of both qubits for the given pair."""
        qubit_dict = {}
        for qubit in ("low", "high"):
            iq = np.column_stack((self[pair][f"{qubit}_i"], self[pair][f"{qubit}_q"]))
            signal = (
                PCA().fit_transform(iq)[:, 0]
                if len(iq) and np.isfinite(iq).all() and np.any(iq != iq[0])
                else np.zeros(len(iq))
            )
            # PCA signs are arbitrary and opposite readout responses must not cancel.
            if qubit_dict and np.dot(next(iter(qubit_dict.values())), signal) < 0:
                signal = -signal
            qubit_dict[qubit] = signal

        return qubit_dict["low"], qubit_dict["high"]


@dataclass
class AvoidedCrossingResults(Results):
    """Estimated crossings, each refined by RANSAC on its own bias interval."""

    crossings: dict[QubitPairId, list[dict]] = field(default_factory=dict)
    """Each crossing stores ``bias`` [a.u.], ``frequency`` [Hz], sloped-asymptote
    ``slope`` [Hz/a.u.], ``coupling`` [Hz], the ``bias_range`` [a.u.] of the peaks
    assigned to it, and whether RANSAC ``fitted`` it. Unfitted crossings keep their
    estimate, with the median frequency and coupling of all estimates.
    The two asymptotes are f = frequency and f = frequency + slope*(x - bias)."""
    peak_biases: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """All extracted peak biases [a.u.] for each pair's mean readout map."""
    peak_frequencies: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """All extracted peak frequencies [Hz] for each pair's mean readout map."""
    inliers: dict[QubitPairId, list] = field(default_factory=dict)
    """Boolean RANSAC inlier mask over the full extracted peak set, combining the
    fits of all crossings. Present only if at least one crossing was fitted."""
    coupler: dict[QubitPairId, dict] = field(default_factory=dict)
    """Parabola ``coefficients`` ``[c2, c1, c0]`` with their ``errors``, the vertex
    ``sweetspot`` [a.u.] and ``frequency`` [Hz], and ``dof``.
    Errors describe the scatter of the constraints, not measurement uncertainties."""

    def __contains__(self, key):
        # Crossings may exist without a coupler fit, and are still worth reporting.
        return tuple(key) in self.crossings


def _acquisition(
    params: AvoidedCrossingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> AvoidedCrossingData:
    """Probe each qubit of each pair in a separate acquisition."""

    drive_pulse = Pulse(
        duration=params.pulse_duration,
        amplitude=params.pulse_amplitude,
        envelope=Rectangular(),
    )

    flux_pulse = Pulse(
        duration=params.pulse_duration,
        amplitude=to_range(params.bias)[0],
        envelope=GaussianSquare(sigma=0.2, risefall=0),
    )

    full_sequence = PulseSequence()
    updates: dict = {}
    low_ro_pulses_id: dict[QubitPairId, PulseId] = {}
    high_ro_pulses_id: dict[QubitPairId, PulseId] = {}
    drive_sweepers: dict[QubitPairId, Sweeper] = {}
    for pair in targets:
        low_freq_q, high_freq_q = order_pair(pair, platform)

        high_freq_natives = platform.natives.single_qubit[high_freq_q]
        low_freq_natives = platform.natives.single_qubit[low_freq_q]

        low_qd_channel, _ = low_freq_natives.RX()[0]
        low_ro_channel, low_ro_pulse = low_freq_natives.MZ()[0]
        low_center = platform.config(low_qd_channel).frequency
        low_ro_pulses_id[pair] = low_ro_pulse.id

        high_ro_channel, high_ro_pulse = high_freq_natives.MZ()[0]
        high_flux_channel = platform.qubits[high_freq_q].flux
        assert high_flux_channel is not None, "High flux channel must exist"
        high_ro_pulses_id[pair] = high_ro_pulse.id

        # Drive and coupler pulse play together; the readout waits for both to end.
        sequence = PulseSequence(
            [(low_qd_channel, drive_pulse), (high_flux_channel, flux_pulse)]
        )

        # Append the readout pulses to the sequence
        sequence |= PulseSequence(
            [(low_ro_channel, low_ro_pulse), (high_ro_channel, high_ro_pulse)]
        )

        full_sequence += sequence

        drive_sweepers[pair] = Sweeper(
            parameter=Parameter.frequency,
            range=to_range(params.frequency, low_center),
            channels=[low_qd_channel],
        )

        updates |= {
            platform.qubits[low_freq_q].probe: {
                "frequency": readout_frequency(low_freq_q, platform)
            },
            platform.qubits[high_freq_q].probe: {
                "frequency": readout_frequency(high_freq_q, platform)
            },
        }

    flux_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=to_range(params.bias),
        pulses=[flux_pulse],
    )

    results = platform.execute(
        [full_sequence],
        [[flux_sweeper], list(drive_sweepers.values())],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = AvoidedCrossingData()
    flux_array = np.array(flux_sweeper.values)
    for pair in targets:
        freq = np.array(drive_sweepers[pair].values)
        low_freq_q, high_freq_q = pair
        data.register_qubit(
            AvoidedCrossingType,
            pair,
            {
                "freq": np.tile(freq, len(flux_array)),
                "bias": np.repeat(flux_array, len(freq)),
                "low_i": results[low_ro_pulses_id[pair]][..., 0].ravel(),
                "low_q": results[low_ro_pulses_id[pair]][..., 1].ravel(),
                "high_i": results[high_ro_pulses_id[pair]][..., 0].ravel(),
                "high_q": results[high_ro_pulses_id[pair]][..., 1].ravel(),
            },
        )

    return data


def _extract_peak_coordinates(
    frequencies: npt.NDArray[np.floating],
    biases: npt.NDArray[np.floating],
    signal: npt.NDArray[np.floating],
    max_peaks: int | None = 1,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extract prominent peaks and dips in the qubit (flux, frequency) landscape.

    Keep at most ``max_peaks`` features per bias bin, or all significant features
    when it is ``None``. The default retains only the strongest feature.
    """
    if max_peaks is not None and max_peaks < 1:
        raise ValueError("max_peaks must be positive or None.")

    filtered_signal = filter_data(signal)

    mad = np.median(
        np.abs(filtered_signal - np.median(filtered_signal, axis=1, keepdims=True))
    )
    height_limit = 3.5 * mad

    peak_biases, peak_frequencies = [], []
    for bias, row in zip(biases, filtered_signal):
        # Use find_peaks instead of argmax because there may be nothing in a row. Try
        # both peak and dip per row, since this may differ per row due to moving of the
        # resonator frequency.
        median_subtracted = row - np.median(row)
        peaks, peak_props = find_peaks(
            median_subtracted, height=height_limit, prominence=0
        )
        dips, dip_props = find_peaks(
            -median_subtracted, height=height_limit, prominence=0
        )
        if len(peaks) == 0 and len(dips) == 0:
            continue

        features = np.concatenate((peaks, dips))
        prominence = np.concatenate(
            (peak_props["prominences"], dip_props["prominences"])
        )
        order = np.argsort(-prominence, kind="stable")
        selected = np.sort(features[order[:max_peaks]])
        peak_biases.extend([bias] * len(selected))
        peak_frequencies.extend(frequencies[selected])

    return np.asarray(peak_biases), np.asarray(peak_frequencies)


def _numerical_slope(x, y):
    """Median finite difference over several bins to reduce frequency quantization."""
    step = min(3, len(x) - 1)
    return float(np.median((y[step:] - y[:-step]) / (x[step:] - x[:-step])))


def _crossings(bias, frequencies, signal, extracted_bias, extracted_frequency):
    """Estimate crossings from the already-extracted spectral peaks."""
    if len(bias) < 7 or len(frequencies) < 5:
        return []
    df = float(np.median(np.diff(frequencies)))
    branches = np.full((len(bias), 2), np.nan)
    bright = np.full(len(bias), np.nan)
    for index, row in enumerate(signal):
        detected = extracted_frequency[extracted_bias == bias[index]]
        contrast = np.abs(row - np.median(row))
        differences = np.diff(row)
        noise = 1.4826 * np.median(np.abs(differences - np.median(differences)))
        strength = np.interp(detected, frequencies, contrast)
        supported = strength > max(5 * noise, 0.05 * np.max(contrast))
        detected, strength = detected[supported], strength[supported]
        if len(detected):
            bright[index] = detected[np.argmax(strength)]
        if len(detected) >= 2:
            branches[index] = np.sort(detected[np.argsort(strength)[-2:]])

    estimates = []
    resolved = np.flatnonzero(np.isfinite(branches).all(axis=1))
    for segment in np.split(resolved, np.flatnonzero(np.diff(resolved) > 1) + 1):
        if len(segment) < 7:
            continue
        gaps = np.diff(branches[segment], axis=1).ravel()
        gaps = median_filter(gaps, size=3, mode="nearest")
        minima, _ = find_peaks(
            -gaps, prominence=max(3 * df, 0.15 * np.ptp(gaps)), distance=3
        )
        for minimum in minima:
            start = max(0, minimum - WINDOW)
            stop = min(len(segment), minimum + WINDOW + 1)
            if minimum - start < 2 or stop - minimum < 3:
                continue
            indices = segment[start:stop]
            midpoint = np.mean(branches[segment[minimum]])
            slope = _numerical_slope(bias[indices], branches[indices].sum(axis=1))
            if abs(slope) * np.ptp(bias[indices]) < 3 * df:
                continue
            estimates.append(
                {
                    "bias": float(bias[segment[minimum]]),
                    "frequency": float(midpoint),
                    "slope": slope,
                    "coupling": float(np.diff(branches[segment[minimum]])[0] / 2),
                }
            )

    detected = np.isfinite(bright)
    peak_bias, peak_frequency = bias[detected], bright[detected]
    if len(peak_bias) < 7:
        return sorted(estimates, key=lambda crossing: crossing["bias"])
    trace = median_filter(peak_frequency, size=3, mode="nearest")
    jumps = np.diff(trace)
    typical = np.median(np.abs(jumps))
    scatter = 1.4826 * np.median(np.abs(np.abs(jumps) - typical))
    candidates, _ = find_peaks(
        np.abs(jumps), height=max(3 * df, typical + 6 * scatter), distance=3
    )
    for position, index in enumerate(candidates):
        if peak_bias[index + 1] - peak_bias[index] > 3 * np.median(np.diff(bias)):
            continue
        crossing_bias = float(np.mean(peak_bias[index : index + 2]))
        if any(
            abs(crossing_bias - crossing["bias"]) <= WINDOW * np.median(np.diff(bias))
            for crossing in estimates
        ):
            continue
        left = candidates[position - 1] + 1 if position else 0
        right = (
            candidates[position + 1] + 1
            if position + 1 < len(candidates)
            else len(trace)
        )
        start, stop = max(left, index - 3), min(right, index + 5)
        if index - start < 2 or stop - index < 4:
            continue
        slope = _numerical_slope(
            peak_bias[start : index + 1], peak_frequency[start : index + 1]
        ) + _numerical_slope(
            peak_bias[index + 1 : stop], peak_frequency[index + 1 : stop]
        )
        if (
            slope * jumps[index] >= 0
            or abs(slope) * np.ptp(peak_bias[start:stop]) < 3 * df
        ):
            continue
        estimates.append(
            {
                "bias": crossing_bias,
                "frequency": float(np.mean(peak_frequency[index : index + 2])),
                "slope": slope,
                "coupling": float(
                    abs(np.diff(peak_frequency[index : index + 2])[0]) / 2
                ),
            }
        )
    return sorted(estimates, key=lambda crossing: crossing["bias"])


def _avoided_crossing(bias, crossing_bias, frequency, slope, coupling, branch):
    """Eigenfrequencies of a horizontal qubit coupled to a locally linear qubit."""
    detuning = slope * (np.asarray(bias) - crossing_bias)
    return frequency + detuning / 2 + branch * np.hypot(detuning / 2, coupling)


def _fit_crossing(x, y, seed, frequencies):
    """RANSAC-fit eigenbranches using every supplied peak across the full sweep."""

    threshold = adaptive_residual_threshold(0.5e6, frequencies)

    domain = np.asarray([np.min(x), np.max(x)])
    frequency_scale = max(seed["coupling"], threshold)
    bias_scale = np.ptp(domain) / 2
    centerline = seed["frequency"] + seed["slope"] * (x - seed["bias"]) / 2
    branches = np.where(y >= centerline, 1, -1)

    normalized_x = (x - seed["bias"]) / bias_scale
    normalized_y = (y - seed["frequency"]) / frequency_scale
    slope = seed["slope"] * bias_scale / frequency_scale

    # Scalar point indices keep the shared RANSAC continuity score ordered by bias,
    # while allowing two branch observations at the same bias.
    def model(index, center, offset, gradient, coupling):
        index = np.asarray(index, dtype=int)
        return _avoided_crossing(
            normalized_x[index], center, offset, gradient, coupling, branches[index]
        )

    slope_bounds = sorted((0.2 * slope, 3 * slope))
    bounds = (
        [
            (domain[0] - seed["bias"]) / bias_scale,
            -1.0,
            slope_bounds[0],
            max(threshold / 2, seed["coupling"] / 4) / frequency_scale,
        ],
        [
            (domain[1] - seed["bias"]) / bias_scale,
            1.0,
            slope_bounds[1],
            3.0,
        ],
    )
    parameters, inliers = ransac_fit(
        np.arange(len(x), dtype=float),
        normalized_y,
        fit_function=model,
        residual_threshold=threshold / frequency_scale,
        bounds=bounds,
        max_trials=500,
    )
    center, offset, gradient, coupling = parameters
    crossing_bias = float(seed["bias"] + center * bias_scale)
    return {
        "bias": crossing_bias,
        "frequency": float(seed["frequency"] + offset * frequency_scale),
        "slope": float(gradient * frequency_scale / bias_scale),
        "coupling": float(coupling * frequency_scale),
        "inliers": np.asarray(inliers, dtype=bool).tolist(),
    }


def _fit(data: AvoidedCrossingData) -> AvoidedCrossingResults:
    """Extract once, estimate crossings, and fit each on its own bias interval."""
    results = AvoidedCrossingResults()
    for pair in data.data:
        bias, frequencies = data.bias, data.pair_frequency(pair)

        low, high = data.pair_signal(pair)
        avg_signal = np.mean([low, high], axis=0)
        avg_signal = avg_signal.reshape(len(bias), len(frequencies))
        peaks_biases, peaks_freqs = _extract_peak_coordinates(
            frequencies=frequencies, biases=bias, signal=avg_signal, max_peaks=2
        )
        results.peak_biases[pair] = peaks_biases.tolist()
        results.peak_frequencies[pair] = peaks_freqs.tolist()
        try:
            estimates = _crossings(
                bias, frequencies, avg_signal, peaks_biases, peaks_freqs
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            log.warning("Avoided-crossing analysis failed for pair %s: %s", pair, error)
            continue

        if not estimates:
            log.warning("No resolved avoided crossing for pair %s.", pair)
            continue

        median_frequency = float(np.median([c["frequency"] for c in estimates]))
        median_coupling = float(np.median([c["coupling"] for c in estimates]))
        seeds = [
            crossing | {"frequency": median_frequency, "coupling": median_coupling}
            for crossing in estimates
        ]

        # Split the bias axis halfway between adjacent crossings; the outer
        # crossings also take the extreme bias intervals. Peaks exactly on a
        # boundary go to the left crossing.
        seed_biases = np.array([seed["bias"] for seed in seeds])
        boundaries = (seed_biases[1:] + seed_biases[:-1]) / 2
        edges = np.concatenate(([bias[0]], boundaries, [bias[-1]]))
        owner = np.searchsorted(boundaries, peaks_biases, side="left")

        crossings, inliers = [], np.zeros(len(peaks_biases), dtype=bool)
        for index, seed in enumerate(seeds):
            bias_range = [float(edges[index]), float(edges[index + 1])]
            selected = np.flatnonzero(owner == index)
            crossing = seed | {"bias_range": bias_range}
            try:
                fit_hyperbola = _fit_crossing(
                    peaks_biases[selected],
                    peaks_freqs[selected],
                    seed,
                    frequencies,
                )
                inliers[selected] = fit_hyperbola.pop("inliers")
                crossing |= fit_hyperbola
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                log.warning(
                    "RANSAC failed for pair %s near bias %.6g: %s",
                    pair,
                    seed["bias"],
                    error,
                )
            crossings.append(crossing)

        # if the fit fails we save the first estimation of the crossing without the fitted hyperbola
        results.crossings[pair] = crossings
        # if the fit succeeds at least for one crossing, we save the inliers
        if inliers.any():
            results.inliers[pair] = inliers.tolist()

    return results


def _plot_crossings(
    crossing: dict,
    freq_limit: tuple[float, float],
    number: int = 0,
) -> list[go.Scatter]:

    scatters: list[go.Scatter] = []
    color = qualitative.Plotly[(number - 1) % len(qualitative.Plotly)]
    low, high = crossing["bias_range"]
    bias_points = np.linspace(low, high, 100)
    for slope in (0, crossing["slope"]):
        asymptote = crossing["frequency"] + slope * (bias_points - crossing["bias"])
        asymptote_mask = (asymptote >= freq_limit[0]) & (asymptote <= freq_limit[1])
        scatters.append(
            go.Scatter(
                x=bias_points[asymptote_mask],
                y=asymptote[asymptote_mask] * scipy.constants.nano,
                mode="lines",
                line={"dash": "dash", "color": color},
                legendgroup=f"Crossing {number}",
                showlegend=False,
            ),
        )
    scatters.append(
        go.Scatter(
            x=[crossing["bias"]],
            y=[crossing["frequency"] * scipy.constants.nano],
            mode="markers",
            marker={"symbol": "x", "size": 12, "color": color},
            name=f"Crossing {number}",
            legendgroup=f"Crossing {number}",
        ),
    )
    return scatters


def _plot_hyperbolas(
    crossing: dict,
    signal_bias: npt.NDArray[np.float64],
    signal_freq: npt.NDArray[np.float64],
    inliers: npt.NDArray[np.bool],
    freq_limit: tuple[float, float],
    number: int = 0,
) -> list[go.Scatter]:
    """Inliers and fitted eigenbranches over the crossing's own bias interval."""

    color = qualitative.Plotly[(number - 1) % len(qualitative.Plotly)]
    low, high = crossing["bias_range"]
    owned = inliers & (signal_bias >= low) & (signal_bias <= high)
    bias_points = np.linspace(low, high, 100)
    scatters_list: list[go.Scatter] = []
    scatters_list.append(
        go.Scatter(
            x=signal_bias[owned],
            y=signal_freq[owned] * scipy.constants.nano,
            mode="markers",
            marker={
                "color": color,
                "size": 7,
                "line": {"color": "black", "width": 1},
            },
            name="RANSAC inliers",
            legendgroup="RANSAC inliers",
            showlegend=number == 1,
        )
    )
    for label, branch in (("Lower", -1), ("Upper", 1)):
        freqs_plot = _avoided_crossing(
            bias_points,
            crossing["bias"],
            crossing["frequency"],
            crossing["slope"],
            crossing["coupling"],
            branch,
        )
        hyp_mask = (freqs_plot >= freq_limit[0]) & (freqs_plot <= freq_limit[1])
        scatters_list.append(
            go.Scatter(
                x=bias_points[hyp_mask],
                y=freqs_plot[hyp_mask] * scipy.constants.nano,
                mode="lines",
                line={"color": color},
                name=(
                    "Fitted hyperbolas" if branch == -1 else f"{label} fitted branch"
                ),
                legendgroup="Fitted hyperbolas",
                showlegend=branch == -1 and number == 1,
            )
        )
    return scatters_list


def _plot(
    data: AvoidedCrossingData,
    target: QubitPairId,
    fit: AvoidedCrossingResults | None = None,
):
    """Plot the mean above both readouts, with shared overlay toggles."""
    bias = data.bias
    q_freqs = data.pair_frequency(target)
    bias_limits = (bias[0], bias[-1])
    low, high = data.pair_signal(target)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Mean of the two qubit readouts",
            "Low-frequency qubit readout",
            "High-frequency qubit readout",
        ),
        horizontal_spacing=0.18,
        vertical_spacing=0.18,
    )
    for readout, signal, row, column in (
        ("mean", np.mean([low, high], axis=0), 1, 1),
        ("low", low, 2, 1),
        ("high", high, 2, 2),
    ):
        figure.add_trace(
            go.Heatmap(
                x=bias,
                y=q_freqs * scipy.constants.nano,
                z=signal.reshape(len(bias), len(q_freqs)).T,
                name=f"{readout} readout",
                colorbar={
                    "title": {"text": "Signal [a.u.]", "side": "right"},
                    "x": 0.43 if readout == "low" else 1.02,
                    "y": 0.795 if readout == "mean" else 0.205,
                    "len": 0.41,
                    "thickness": 12,
                },
                colorscale="Inferno",
            ),
            row=row,
            col=column,
        )

        if readout == "mean" and fit is not None and target in fit:
            figure.add_trace(
                go.Scatter(
                    x=fit.peak_biases[target],
                    y=np.asarray(fit.peak_frequencies[target]) * scipy.constants.nano,
                    mode="markers",
                    marker={
                        "color": "white",
                        "size": 5,
                        "line": {"color": "black", "width": 1},
                    },
                    name="Extracted signal",
                    legendgroup="extracted signal",
                ),
                row=row,
                col=column,
            )

            for number, crossing in enumerate(fit.crossings[target], start=1):
                figure.add_traces(
                    _plot_crossings(
                        crossing=crossing,
                        freq_limit=(q_freqs[0], q_freqs[-1]),
                        number=number,
                    ),
                    rows=row,
                    cols=column,
                )
                if target in fit.inliers:
                    figure.add_traces(
                        _plot_hyperbolas(
                            crossing=crossing,
                            signal_bias=np.asarray(fit.peak_biases[target]),
                            signal_freq=np.asarray(fit.peak_frequencies[target]),
                            inliers=np.asarray(fit.inliers[target], dtype=bool),
                            freq_limit=(q_freqs[0], q_freqs[-1]),
                            number=number,
                        ),
                        rows=row,
                        cols=column,
                    )

    figure.update_xaxes(
        title_text="High freq qubit bias [a.u.]",
        range=bias_limits,
        autorange=False,
    )
    figure.update_yaxes(
        title_text="Low freq qubit drive frequency [GHz]",
        range=(q_freqs[0] * scipy.constants.nano, q_freqs[-1] * scipy.constants.nano),
        autorange=False,
    )
    figure.update_layout(
        height=900,
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.2,
            "yanchor": "top",
            "itemclick": "toggle",
            "groupclick": "togglegroup",
        },
        margin={"b": 160},
    )

    report = ""
    if fit is not None and target in fit:
        crossings = fit.crossings[target]
        report = table_html(
            {
                "Crossing": list(range(1, len(crossings) + 1)),
                "Bias [a.u.]": [c["bias"] for c in crossings],
                "Frequency [GHz]": [
                    c["frequency"] * scipy.constants.nano for c in crossings
                ],
                "Coupling [MHz]": [
                    c.get("coupling", np.nan) * scipy.constants.micro for c in crossings
                ],
                "Bias range [a.u.]": [
                    f"[{c['bias_range'][0]:.4g}, {c['bias_range'][1]:.4g}]"
                    for c in crossings
                ],
            }
        )
    return [figure], report


def _update(
    results: AvoidedCrossingResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    """Store the median pair coupling [Hz], preferring RANSAC-fitted crossings."""
    crossings = results.crossings.get(target, [])
    if crossings:
        coupling = float(np.median([c["coupling"] for c in crossings]))
        update.pair_coupling([coupling, 0.0], platform, target)


avoided_crossing = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""AvoidedCrossing Routine object."""
