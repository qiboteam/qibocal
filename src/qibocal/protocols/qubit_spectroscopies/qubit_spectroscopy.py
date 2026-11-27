from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ChannelId,
    Delay,
    IqConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    lorentzian,
    table_dict,
    table_html,
    to_range,
)
from qibocal.result import magnitude, phase

__all__ = [
    "QubitSpectroscopyData",
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "_fit",
    "qubit_spectroscopy",
]

BATCH_OVERLAP = 10
"""Number of points shared by adjacent batches, used to stitch them."""


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    frequency: RangeLike | None = None
    """Drive frequency [Hz] range for sweep."""
    freq_width: int | None = None
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int | None = None
    """Frequency [Hz] step for sweep."""
    drive_duration: int = 4000
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: float = 1.0
    """Drive pulse amplitude (optional). Same for all qubits."""

    def frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            assert self.freq_width is not None and self.freq_step is not None
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )

    def f0(self, center: float = 0.0) -> float:
        """Return the center frequency of the sweep."""
        start, end, _ = self.frequency_range(center)
        return (start + end) / 2


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    lo_frequencies: dict[QubitId, list[float | None]] = field(default_factory=dict)
    """LO frequency [Hz] used in each batch."""
    steps: dict[QubitId, list[float]] = field(default_factory=dict)
    """Signal step [a.u.] of each batch with respect to its inner neighbour."""
    flagged: dict[QubitId, list[bool]] = field(default_factory=dict)
    """Whether the step of each batch is significant."""


@dataclass
class QubitSpectroscopyData(Data):
    """QubitSpectroscopy acquisition outputs."""

    drive_frequencies: dict[QubitId, list[float]]
    """Frequencies."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[np.float64]]
    """Raw data acquired."""
    batch_indices: dict[QubitId, list[int]] = field(default_factory=dict, kw_only=True)
    """Batch index of every point."""
    lo_frequencies: dict[QubitId, list[float | None]] = field(
        default_factory=dict, kw_only=True
    )
    """LO frequency [Hz] used in each batch."""
    anchor_batch: dict[QubitId, int] = field(default_factory=dict, kw_only=True)
    """Index of the batch the others are aligned to."""

    def signal(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return magnitude(self.data[qubit])

    def phase(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return phase(self.data[qubit])


@dataclass
class _Batch:
    """Frequencies of a single qubit swept in a single execution."""

    frequencies: npt.NDArray[np.float64]
    """Absolute drive frequencies [Hz]."""
    lo_frequency: float | None
    """LO frequency [Hz] used to sweep :attr:`frequencies`."""
    anchor: bool
    """Whether the other batches are aligned to this one."""


def _split_outward(
    outward: npt.NDArray[np.int64], inner: npt.NDArray[np.int64], max_points: int
) -> list[npt.NDArray[np.int64]]:
    """Split the ``outward`` indices in equal batches, moving away from ``inner``.

    Both ``outward`` and ``inner`` are ordered in the outward direction. Each batch
    also includes up to :data:`BATCH_OVERLAP` points of its inner neighbour, and
    contains at most ``max_points`` points.
    """
    if len(outward) == 0:
        return []
    overlap = min(BATCH_OVERLAP, max_points // 2)
    nbatches = int(np.ceil(len(outward) / (max_points - overlap)))
    batches = []
    for chunk in np.array_split(outward, nbatches):
        inner = np.concatenate([inner[max(len(inner) - overlap, 0) :], chunk])
        batches.append(inner)
    return batches


def _plan_batches(
    frequencies: npt.NDArray[np.float64],
    lo_frequency: float | None,
    max_if_bandwidth: float = 300e6,
) -> list[_Batch]:
    """Split the sweep of a qubit in batches, sorted by frequency.

    The anchor batch contains the frequencies within ``max_if_bandwidth`` from the
    current LO, which is left untouched. The remaining frequencies are split in
    equal batches moving outward from the anchor, each with its LO at its centre.
    If no frequency is close enough to the current LO, the anchor is the batch
    nearest to it, with its LO moved as well.
    """
    if lo_frequency is None:
        return [_Batch(frequencies, None, True)]

    indices = np.arange(len(frequencies))

    # largest number of points spanning at most twice the IF bandwidth
    max_points = int(
        np.searchsorted(
            frequencies, frequencies[0] + 2 * max_if_bandwidth, side="right"
        )
    )
    window = np.abs(frequencies - lo_frequency) <= max_if_bandwidth
    anchor_lo = lo_frequency
    if window.any():
        anchor = indices[window]
    else:
        anchor = (
            indices[:max_points]
            if frequencies[0] > lo_frequency
            else indices[-max_points:]
        )
        anchor_lo = float(frequencies[anchor[0]] + frequencies[anchor[-1]]) / 2

    lower = _split_outward(indices[: anchor[0]][::-1], anchor[::-1], max_points)
    upper = _split_outward(indices[anchor[-1] + 1 :], anchor, max_points)
    outer = [frequencies[np.sort(batch)] for batch in lower[::-1] + upper]
    batches = [_Batch(freqs, float(freqs[0] + freqs[-1]) / 2, False) for freqs in outer]
    batches.insert(len(lower), _Batch(frequencies[anchor], anchor_lo, True))
    return batches


def _acquisition(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by splitting the sweep of each qubit in batches,
    each within the IF bandwidth of its LO.
    """

    drive_channels: dict[QubitId, ChannelId] = {}
    lo_channels: dict[QubitId, ChannelId | None] = {}
    lo_frequencies: dict[QubitId, float | None] = {}
    batches: dict[QubitId, list[_Batch]] = {}
    for qubit in targets:
        drive_channel = platform.qubits[qubit].drive
        assert drive_channel is not None
        drive_channels[qubit] = drive_channel
        lo = lo_channels[qubit] = platform.channels[drive_channel].lo
        lo_frequencies[qubit] = (
            platform.config(lo).frequency if lo is not None else None
        )
        frequencies = np.arange(
            *params.frequency_range(
                cast(IqConfig, platform.config(drive_channel)).frequency
            )
        )
        batches[qubit] = _plan_batches(frequencies, lo_frequencies[qubit])
        if any(
            b.anchor and b.lo_frequency != lo_frequencies[qubit] for b in batches[qubit]
        ):
            log.warning(
                f"No frequency swept for qubit {qubit} lies within the IF bandwidth "
                f"of {lo}: no batch runs on a calibrated LO."
            )

    # moving an LO shared by multiple qubits would set it to multiple values at once
    shared_los = defaultdict(list)
    for qubit, lo in lo_channels.items():
        if lo is not None:
            shared_los[lo].append(qubit)
    for lo, qubits in shared_los.items():
        if len(qubits) > 1 and any(
            b.lo_frequency != lo_frequencies[q] for q in qubits for b in batches[q]
        ):
            raise NotImplementedError(
                f"Qubits {qubits} share the drive LO {lo}, which would need to be moved "
                "to sweep the requested frequencies."
            )

    # Initialize storage for intermediate results
    values = {qubit: defaultdict(list) for qubit in targets}
    # TODO: remove, and propagate differently, since it is always the same for
    # all qubits
    drive_amplitudes: dict[QubitId, float] = {
        q: params.drive_amplitude for q in targets
    }

    # parallel sweepers are iterated together, thus they need the same length:
    # execution ``k`` is split by the length of the batches
    executions = defaultdict(list)
    for qubit in targets:
        for k, batch in enumerate(batches[qubit]):
            executions[k, len(batch.frequencies)].append(qubit)

    # Execute each batch
    for (k, _), qubits in sorted(executions.items()):
        # Build the pulse sequence
        sequence = PulseSequence()
        ro_pulses = {}
        sweepers = []
        updates = {}

        for qubit in qubits:
            natives = platform.natives.single_qubit[qubit]
            qd_channel = drive_channels[qubit]
            assert natives.MZ is not None
            [(ro_channel, ro_pulse)] = natives.MZ()

            qd_pulse = Pulse(
                amplitude=params.drive_amplitude,
                duration=params.drive_duration,
                envelope=Rectangular(),
            )
            ro_pulses[qubit] = ro_pulse

            sequence.append((qd_channel, qd_pulse))
            sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((ro_channel, ro_pulse))

            batch = batches[qubit][k]
            sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=batch.frequencies,
                    channels=[qd_channel],
                )
            )

            # move the LO, together with the drive channel to pass qibolab's IF validation
            if batch.lo_frequency != lo_frequencies[qubit]:
                updates[lo_channels[qubit]] = {"frequency": batch.lo_frequency}
                updates[qd_channel] = {"frequency": batch.lo_frequency}

        # Execute this batch
        results = platform.execute(
            [sequence],
            [sweepers],
            updates=[updates],
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # Collect results from this batch
        for qubit in qubits:
            frequencies = batches[qubit][k].frequencies

            # Store results with absolute frequencies
            values[qubit]["frequency"].append(frequencies)
            values[qubit]["result"].append(results[ro_pulses[qubit].id])
            values[qubit]["batch"].append(np.full(len(frequencies), k))

    freqs: dict[QubitId, list[float]] = {}
    data: dict[QubitId, npt.NDArray[np.float64]] = {}
    batch_indices: dict[QubitId, list[int]] = {}
    # Combine all batches for each qubit
    for q in targets:
        # Concatenate arrays from all batches
        freqs[q] = np.concatenate(values[q]["frequency"]).tolist()
        data[q] = np.concatenate(values[q]["result"])
        batch_indices[q] = np.concatenate(values[q]["batch"]).tolist()

    return QubitSpectroscopyData(
        drive_frequencies=freqs,
        amplitudes=drive_amplitudes,
        data=data,
        batch_indices=batch_indices,
        lo_frequencies={q: [b.lo_frequency for b in batches[q]] for q in targets},
        anchor_batch={
            q: next(k for k, b in enumerate(batches[q]) if b.anchor) for q in targets
        },
    )


def _lorentzian_with_offset(frequency, amplitude, center, sigma, offset):
    return lorentzian(frequency, amplitude, center, sigma) + offset


def _guess_initial_parameters(signal, frequencies, is_peak):
    k = max(1, int(0.1 * len(signal)))
    guess_offset = np.mean(np.concatenate([signal[:k], signal[-k:]]))

    guess_offset = (signal[0] + signal[-1]) / 2
    signal_no_background = signal - guess_offset

    sign = 1 if is_peak else -1
    signal_flipped = signal_no_background * sign
    guess_center = frequencies[np.argmax(signal_flipped)]
    guess_peak_height = signal_flipped.max() * sign
    indices_beyond_half = np.where(signal_flipped > signal_flipped.max() / 2)[0]

    if len(indices_beyond_half) >= 1:
        guess_sigma = (
            frequencies[indices_beyond_half[-1]] - frequencies[indices_beyond_half[0]]
        ) / 2
    else:
        # if there is no clear peak, we give a high flexibility
        guess_sigma = frequencies[-1] - frequencies[0]

    guess_amp = guess_peak_height * guess_sigma * np.pi

    return [
        guess_amp,
        guess_center,
        guess_sigma,
        guess_offset,
    ]


def _lorentzian_fit(
    frequencies: np.ndarray, signal: np.ndarray
) -> tuple[float, list[float], list[float]]:
    freq_domain_size = frequencies[-1] - frequencies[0]

    # Try to fit both peak and dip, and pick the best one
    best_fit = None
    for is_peak in [True, False]:
        initial_parameters = _guess_initial_parameters(signal, frequencies, is_peak)
        bounds = (
            [0.0 if is_peak else -np.inf, frequencies[0], 0.0, -np.inf],
            [np.inf if is_peak else 0.0, frequencies[-1], freq_domain_size, np.inf],
        )

        # fit the model with the data and guessed parameters
        try:
            fit_parameters, parameters_cov = curve_fit(
                _lorentzian_with_offset,
                frequencies,
                signal,
                p0=initial_parameters,
                bounds=bounds,
            )
        except RuntimeError:
            continue

        sum_sq_residuals = float(
            np.sum(
                (signal - _lorentzian_with_offset(frequencies, *fit_parameters)) ** 2
            )
        )

        if best_fit is None or sum_sq_residuals < best_fit["sum_sq_residuals"]:
            best_fit = {
                "fit_parameters": fit_parameters,
                "parameters_cov": parameters_cov,
                "sum_sq_residuals": sum_sq_residuals,
            }

    if best_fit is None:
        raise RuntimeError("fit failed")

    # The output results are stored in a json, but ndarray is not JSON serializable,
    # so the parameters are converted to list.
    parameter_errors = np.sqrt(np.diag(best_fit["parameters_cov"])).tolist()
    model_parameters = best_fit["fit_parameters"].tolist()
    return (
        model_parameters[1],
        model_parameters,
        parameter_errors,
    )


def _batch_step(
    frequencies: npt.NDArray[np.float64],
    signal: npt.NDArray[np.complex128],
    inner: npt.NDArray[np.bool_],
    outer: npt.NDArray[np.bool_],
) -> tuple[complex, float]:
    """Signal step of the ``outer`` batch with respect to the ``inner`` one.

    The step is the mean complex difference over the shared frequencies, returned
    together with its standard error.
    """
    shared = np.isin(
        frequencies, np.intersect1d(frequencies[inner], frequencies[outer])
    )
    difference = signal[outer & shared] - signal[inner & shared]
    return difference.mean(), difference.std(ddof=1) / np.sqrt(len(difference))


def _stitch(
    data: QubitSpectroscopyData, qubit: QubitId, step_threshold: float = 3
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[float], list[bool]]:
    """Align the batches of ``qubit`` to its anchor batch, and merge them.

    The step between adjacent batches is chained outward from the anchor, and
    subtracted in IQ. The points shared by multiple batches are averaged.

    Returns the merged frequencies and IQ signal, together with the step of each
    batch and whether it is larger than ``step_threshold`` times its
    standard error.
    """
    frequencies = np.array(data.drive_frequencies[qubit])
    iq = np.asarray(data.data[qubit])
    if qubit not in data.batch_indices:
        return frequencies, iq, [], []

    batch = np.array(data.batch_indices[qubit])
    anchor = data.anchor_batch[qubit]
    nbatches = len(data.lo_frequencies[qubit])
    signal = iq[..., 0] + 1j * iq[..., 1]

    steps = np.zeros(nbatches, dtype=complex)
    errors = np.zeros(nbatches)
    offsets = np.zeros(nbatches, dtype=complex)

    # moving outward from the anchor, the inner neighbour is always aligned first
    for k in [*range(anchor + 1, nbatches), *range(anchor - 1, -1, -1)]:
        inner = k - 1 if k > anchor else k + 1
        steps[k], errors[k] = _batch_step(
            frequencies, signal, batch == inner, batch == k
        )
        offsets[k] = offsets[inner] + steps[k]

    # average the corrected points shared by adjacent batches
    merged, inverse, counts = np.unique(
        frequencies, return_inverse=True, return_counts=True
    )
    stitched = np.zeros(len(merged), dtype=complex)
    np.add.at(stitched, inverse, signal - offsets[batch])
    stitched /= counts
    return (
        merged,
        np.stack([stitched.real, stitched.imag], axis=-1),
        np.abs(steps).tolist(),
        (np.abs(steps) > step_threshold * errors).tolist(),
    )


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency: dict[QubitId, float] = {}
    params: dict[QubitId, list[float]] = {}
    steps: dict[QubitId, list[float]] = {}
    flagged: dict[QubitId, list[bool]] = {}
    for qubit in qubits:
        frequencies, iq, steps[qubit], flagged[qubit] = _stitch(data, qubit)
        fit_result = _lorentzian_fit(frequencies, magnitude(iq))
        frequency[qubit], params[qubit], _ = fit_result
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=params,
        amplitude=data.amplitudes,
        lo_frequencies={q: data.lo_frequencies.get(q, []) for q in qubits},
        steps=steps,
        flagged=flagged,
    )


def _plot(data: QubitSpectroscopyData, target: QubitId, fit: QubitSpectroscopyResults):
    """Plot QubitSpectroscopy.

    A single figure is built with two side-by-side panels: the signal
    magnitude and the signal phase, after stitching the batches. A Lorentzian
    fit is overlaid on the magnitude panel when a fit is available. For sweeps
    acquired in multiple batches, the raw data and the batch boundaries are
    shown as well.
    """
    fitting_report = ""
    frequencies, iq, *_ = _stitch(data, target)

    fig = make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Signal",
            "Phase",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,  # plotting in GHz
            y=magnitude(iq),
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,  # plotting in GHz
            y=phase(iq),
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
            mode="markers",
        ),
        row=1,
        col=2,
    )

    if fit is not None:
        freqrange = np.linspace(
            min(frequencies),
            max(frequencies),
            2 * len(frequencies),
        )
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=freqrange * scipy.constants.nano,  # plotting in GHz
                y=_lorentzian_with_offset(freqrange, *params),
                name="Fit",
                mode="lines",
            ),
            row=1,
            col=1,
        )

        if data.amplitudes[target] is not None:
            labels = ["Qubit Frequency [Hz]", "Drive Amplitude [a.u.]"]
            values = [fit.frequency[target], data.amplitudes[target]]

            fitting_report = table_html(
                table_dict(
                    target,
                    labels,
                    values,
                )
            )

        if len(fit.lo_frequencies[target]) > 0:
            fitting_report += table_html(
                {
                    "Batch": list(range(len(fit.lo_frequencies[target]))),
                    "LO Frequency [Hz]": fit.lo_frequencies[target],
                    "Step [a.u.]": fit.steps[target],
                    "Flagged": fit.flagged[target],
                }
            )

    fig.update_layout(
        showlegend=True,
        yaxis_title="Signal [a.u.]",
        xaxis_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
        xaxis2_title="Frequency [GHz]",
    )

    return [fig], fitting_report


def _update(
    results: QubitSpectroscopyResults, platform: CalibrationPlatform, target: QubitId
):
    frequency = results.frequency[target]
    platform.calibration.single_qubits[target].qubit.frequency_01 = frequency
    drive_channel = platform.qubits[target].drive
    lo = platform.channels[drive_channel].lo
    if lo is not None:
        lo_frequency = platform.config(lo).frequency
        # the planner moves the LO only if the frequency is beyond its IF bandwidth
        [batch] = _plan_batches(np.array([frequency]), lo_frequency)
        if batch.lo_frequency != lo_frequency:
            # move the LO, preserving the current intermediate frequency of the drive
            intermediate_frequency = (
                cast(IqConfig, platform.config(drive_channel)).frequency - lo_frequency
            )
            update.lo_frequency(
                frequency - intermediate_frequency, platform, target, "drive"
            )
            log.warning(
                f"Drive LO {lo} of qubit {target} moved to "
                f"{frequency - intermediate_frequency:.0f} Hz: its mixer calibration is "
                "now stale."
            )
    update.drive_frequency(frequency, platform, target)


qubit_spectroscopy = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
