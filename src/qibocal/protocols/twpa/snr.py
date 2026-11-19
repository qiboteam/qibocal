"""Protocol to evaluate the SNR improvement of the TWPA, over its pump parameters.

The gain of the TWPA alone does not tell whether the readout gets better: it ignores
the noise added by the amplifier, thus a point of larger gain may still give a lower
signal to noise ratio (SNR). The SNR improvement carries over to the assignment
fidelity, and it can be measured without any qubit calibration, since it only needs the
readout resonator probed with the TWPA on and off.

Unlike :mod:`.frequency_offset`, which only needs the averages, taken on board, this
protocol needs the spread of the individual shots. They are downloaded in batches, and
immediately reduced to their statistics, since their number grows as ``nshots *
noffsets * nfrequencies * nprobes``.
"""

from dataclasses import dataclass, field, fields
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode

from ...auto.operation import Parameters, Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import RangeLike, table_dict, table_html, to_range
from .utils import (
    Selected,
    TwpaData,
    edges,
    frequency_sweepers,
    map_figure,
    off_updates,
    offset_sweepers,
    probe_labels,
    probe_updates,
    probes_figure,
    pumps,
    readouts,
    reference,
    select,
    shared_note,
    update_pump,
)


@dataclass
class TwpaSnrParameters(Parameters):
    """Inputs of the TWPA SNR improvement scan."""

    amplitude: RangeLike
    """Range of amplitude (offset) values for TWPA sweep."""
    frequency: RangeLike
    """Range of TWPA frequency values for sweep."""
    probes: list[float] | None = None
    """List of probe frequencies to evaluate (Hz).

    If omitted, a single acquisition run is performed where each target is probed
    exclusively at its own calibrated readout frequency.
    """
    probe_duration: float = 4e3
    """Probe wave duration."""
    probe_amplitude: float = 1.0
    """Probe wave amplitude."""
    off_power: float | None = None
    """Pump power [dB] of the TWPA-off reference, cf. :data:`.utils.PUMP_OFF_POWER`."""
    reference_shots: int | None = None
    """Shots of the TWPA-off reference.

    By default, as many as ``nshots``. The reference is a single point, thus it is
    cheap to acquire it with more shots than the scan, to make it the least
    contribution to the uncertainty of the improvement.
    """
    max_values: int = 2_000_000
    """Maximum number of shots (I/Q pairs) of a target, held by a single execution.

    The pump offsets are split in batches accordingly, such that the acquisition
    memory of the controllers is not exceeded: the full scan needs ``nshots *
    noffsets * nfrequencies`` values for each target.
    """

    def __post_init__(self):
        if self.max_values < 1:
            raise ValueError("The maximum number of shots per execution must be >= 1.")


SnrObjective = Literal["snr", "gain"]
"""Improvement optimized by the fit, aggregated across the probes by their minimum.

- ``snr``: the SNR improvement, i.e. the gain net of the added noise.
- ``gain``: the gain alone, as in :mod:`.frequency_offset`.
"""

OBJECTIVE_LABELS = {
    "snr": "Minimum SNR improvement across probes [dB]",
    "gain": "Minimum TWPA Gain across probes [dB]",
}
"""Labels of the maps aggregated across the probes."""


@dataclass
class TwpaSnrFitParameters:
    """Fit parameters of the TWPA operating point selection."""

    neighbourhood: int = 1
    """Half-width of the neighbourhood of each grid point, in grid steps.

    The operating point maximizes the minimum improvement over the neighbourhood, such
    that it lies within the plateau, away from the over-pumping region. ``0`` selects
    the plain maximum.
    """
    objective: SnrObjective = "snr"
    """Improvement optimized across the probes, cf. :data:`SnrObjective`."""

    def __post_init__(self):
        if self.objective not in get_args(SnrObjective):
            raise ValueError(
                f"Unknown objective '{self.objective}', expected one of "
                f"{get_args(SnrObjective)}."
            )


@dataclass
class TwpaSnrResults(Results):
    frequency: dict[QubitId, float]
    """Pump frequency [Hz]."""
    offset: dict[QubitId, float]
    """Pump offset [a.u.]."""
    improvement: dict[QubitId, float]
    """Improvement [dB] optimized by the fit, aggregated across the probes."""
    power: dict[QubitId, float] = field(default_factory=dict)
    """Pump power [dB], corresponding to the optimal offset."""
    candidate: dict[QubitId, bool] = field(default_factory=dict)
    """Whether the point is only a candidate, rather than the result.

    It is a candidate when it lies at the edge of the scanned ranges, thus the optimum
    may lie outside of them. Candidates are reported, but not applied.
    """
    probe_snr: dict[QubitId, list[float]] = field(default_factory=dict)
    """SNR improvement [dB] of each probe at the selected point."""
    probe_gain: dict[QubitId, list[float]] = field(default_factory=dict)
    """TWPA gain [dB] of each probe at the selected point."""
    probe_noise: dict[QubitId, list[float]] = field(default_factory=dict)
    """Noise added [dB] by the TWPA to each probe at the selected point."""
    objective: SnrObjective = "snr"
    """Improvement optimized by the fit."""

    def __contains__(self, key: QubitId) -> bool:
        """Whether the fit produced a point for `key`."""
        return all(
            key in getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), dict)
            and not f.name.startswith("probe_")
            and f.name != "candidate"
        )

    def is_candidate(self, qubit: QubitId) -> bool:
        """Whether the point of `qubit` is only a candidate."""
        return self.candidate.get(qubit, False)


def statistics(shots: npt.NDArray) -> npt.NDArray:
    """Mean and spread of single shots.

    Args:
        shots: the shots along the first axis, and I/Q along the last one.

    Returns:
        An array without the shots axis, whose last axis has three components: the
        mean I and Q, and the spread of the shots. The latter is the root mean square
        of the standard deviations of I and Q.
    """
    if shots.shape[0] < 2:
        raise ValueError("At least two shots are needed to estimate their spread.")
    mean = np.mean(shots, axis=0)
    spread = np.sqrt(np.mean(np.var(shots, axis=0, ddof=1), axis=-1))
    return np.concatenate([mean, spread[..., None]], axis=-1)


@dataclass
class TwpaSnrData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Statistics of the shots, for each pump offset, frequency and probe.

    The shape is ``(noffsets, nfrequencies, nprobes, 3)``, with the mean I and Q, and
    the spread of the shots on the last axis, cf. :func:`statistics`.
    """
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept."""
    reference_value: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Statistics of the shots with TWPA off, for each probe.

    The shape is ``(nprobes, 3)``, as the last axes of :attr:`data`.
    """

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.reference_array(qubit).shape[0]

    def reference_array(self, qubit: QubitId) -> npt.NDArray:
        """Statistics of the reference shots, with shape ``(nprobes, 3)``."""
        return np.array(self.reference_value[qubit]).reshape(-1, 3)

    def _group(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Statistics with the pump on and off, of the probes on the pump of `qubit`.

        The shapes are ``(noffsets, nfrequencies, nprobes, 3)`` and ``(nprobes, 3)``.
        """
        group = self.group(qubit)
        signal = np.concatenate([self[q] for q in group], axis=2)
        reference = np.concatenate([self.reference_array(q) for q in group])
        if not np.all(np.isfinite(reference)) or np.any(np.isclose(reference[:, 2], 0)):
            raise ValueError(
                f"TWPA-off reference for qubit {qubit} must be finite, with a "
                "nonzero spread, for every probe."
            )
        if np.any(np.isclose(magnitude(reference[:, :2]), 0)):
            raise ValueError(
                f"TWPA-off reference magnitude for qubit {qubit} must be nonzero, "
                "for every probe."
            )
        return signal, reference

    def probe_gain(self, qubit: QubitId) -> npt.NDArray:
        """TWPA gain [dB] of each probe, with respect to its own reference.

        It is the ratio of the magnitudes of the mean signals. The probes are all the
        ones amplified by the pump of `qubit`, as listed by :meth:`group_probes`. The
        shape is ``(noffsets, nfrequencies, nprobes)``.
        """
        signal, reference = self._group(qubit)
        return 20 * np.log10(magnitude(signal[..., :2]) / magnitude(reference[:, :2]))

    def probe_noise(self, qubit: QubitId) -> npt.NDArray:
        """Noise [dB] added by the TWPA to each probe, with the shape of the gain.

        It is the ratio of the spreads of the shots, with and without pump, and it is
        the part of the gain which is not signal.
        """
        signal, reference = self._group(qubit)
        return 20 * np.log10(signal[..., 2] / reference[:, 2])

    def probe_snr(self, qubit: QubitId) -> npt.NDArray:
        """SNR improvement [dB] of each probe, with the shape of the gain.

        The SNR is the magnitude of the mean signal over the spread of the shots, thus
        the improvement is the gain net of the added noise.
        """
        return self.probe_gain(qubit) - self.probe_noise(qubit)

    def improvement(self, qubit: QubitId, objective: SnrObjective) -> npt.NDArray:
        """Improvement [dB] map, as in `objective`, of the worst probe.

        It is the minimum across the probes, since the worst one limits a multiplexed
        readout.
        """
        probes = self.probe_snr(qubit) if objective == "snr" else self.probe_gain(qubit)
        return np.min(probes, axis=2)


def _batches(size: int, rows: int) -> list[slice]:
    """Slices of at most `rows` out of `size` rows."""
    return [slice(start, start + rows) for start in range(0, size, rows)]


def _acquisition(
    params: TwpaSnrParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaSnrData:
    """Acquire the shots statistics with the TWPA off, and over its pump.

    As in :func:`.frequency_offset._acquisition`, a reference with the TWPA off is
    followed by a 2D hardware sweep over the pump frequency and offset, for each probe.
    But the individual shots are acquired, to know their spread: they are reduced to
    their statistics as soon as downloaded. To bound the memory they need, the sweep is
    split in batches of pump offsets, according to ``params.max_values``.
    """
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes)

    freq_ranges = {
        q: to_range(params.frequency, center=pump.config[q].frequency) for q in targets
    }
    freq_sweeps = frequency_sweepers(pump, freq_ranges)
    frequencies = {q: s.values for q, s in zip(pump.unique.values(), freq_sweeps)}
    offsets = offset_sweepers(pump, to_range(params.amplitude))[0].values

    # the shots default to the platform ones, which the batches have to know
    nshots = params.nshots if params.nshots is not None else platform.settings.nshots

    # 1. Reference measurements with TWPA off
    references = reference(
        platform,
        sequence,
        handles,
        updates,
        off_updates(platform, pump.unique, params.off_power),
        params.reference_shots or nshots,
        params.relaxation_time,
        AveragingMode.SINGLESHOT,
    )
    reference_data = {
        q: [statistics(probe[q]).tolist() for probe in references] for q in targets
    }

    # 2. 2D TWPA sweeps (amplitude and frequency), in batches of offsets
    nfrequencies = max(len(f) for f in frequencies.values())
    rows = max(1, params.max_values // (nshots * nfrequencies))
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    for probe in updates:
        batches: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
        for batch in _batches(len(offsets), rows):
            results = platform.execute(
                [sequence],
                [offset_sweepers(pump, offsets[batch]), freq_sweeps],
                nshots=nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.SINGLESHOT,
                updates=[probe],
            )
            for qubit in targets:
                batches[qubit].append(statistics(results[handles[qubit]]))
        for qubit in targets:
            raw_data[qubit].append(np.concatenate(batches[qubit], axis=0))

    data = TwpaSnrData(
        offset={q: offsets.tolist() for q in targets},
        frequency={
            q: frequencies[pump.unique[pump.channel[q]]].tolist() for q in targets
        },
        reference_value=reference_data,
        probes=params.probes,
        power=pump.power,
        pump=pump.channel,
    )
    for qubit in targets:
        data.data[qubit] = np.stack(raw_data[qubit], axis=2)
    return data


def _fit(
    data: TwpaSnrData, fitpars: TwpaSnrFitParameters | None = None
) -> TwpaSnrResults:
    """Select a robust operating point, within the improvement plateau.

    The improvement optimized (by default, the SNR one) is aggregated across the
    probes by its minimum, and the point is selected as in :func:`.utils.select`.
    Targets sharing a pump are amplified by the same working point: the improvement is
    aggregated over all their probes, and the same point is selected for all of them.
    """
    fitpars = TwpaSnrFitParameters() if fitpars is None else fitpars
    frequency, offset, improvement, power, candidate = {}, {}, {}, {}, {}
    probe_snr, probe_gain, probe_noise = {}, {}, {}
    for qubit in data.qubits:
        if qubit in frequency:
            # already selected, together with a target sharing its pump
            continue
        grid = data.improvement(qubit, fitpars.objective)
        i, j, is_candidate = select(grid, fitpars.neighbourhood)
        point = {
            "snr": data.probe_snr(qubit)[i, j].tolist(),
            "gain": data.probe_gain(qubit)[i, j].tolist(),
            "noise": data.probe_noise(qubit)[i, j].tolist(),
        }
        for q in data.group(qubit):
            candidate[q] = is_candidate
            frequency[q] = data.frequency[q][j]
            offset[q] = data.offset[q][i]
            improvement[q] = grid[i, j]
            power[q] = data.power.get(q, 0.0) + 20 * np.log10(abs(offset[q]))
            probe_snr[q], probe_gain[q], probe_noise[q] = (
                point["snr"],
                point["gain"],
                point["noise"],
            )
    return TwpaSnrResults(
        frequency=frequency,
        offset=offset,
        improvement=improvement,
        power=power,
        candidate=candidate,
        probe_snr=probe_snr,
        probe_gain=probe_gain,
        probe_noise=probe_noise,
        objective=fitpars.objective,
    )


def _selected(fit: TwpaSnrResults | None, target: QubitId) -> Selected | None:
    """The point selected by the fit for `target`, if any."""
    if fit is None or target not in fit:
        return None
    return Selected(
        frequency=fit.frequency[target],
        offset=fit.offset[target],
        candidate=fit.is_candidate(target),
    )


def _plot(data: TwpaSnrData, fit: TwpaSnrResults | None, target: QubitId):
    """Plot the SNR improvement and the gain, and report fit results - if any.

    The first figure is the map optimized by the fit (by default, the minimum SNR
    improvement across the probes), over the pump frequency and offset. The other one
    follows, since the two differ by the noise added by the TWPA, and the gain alone
    may mislead the choice of the point. With several probes, their SNR improvements
    close the figures. Maps are plotted only for the first target of a shared pump.
    """
    objective = fit.objective if fit is not None else "snr"
    other = "gain" if objective == "snr" else "snr"
    group = data.group(target)
    shared = shared_note(data, target, "improvement")
    selected = _selected(fit, target)
    frequency, offset = data.frequency[target], data.offset[target]
    base_power = data.power.get(target, 0.0)
    labels = probe_labels(data.probes, group, data.group_probes(target))

    figures = []
    if target == group[0]:
        figures = [
            map_figure(
                frequency,
                offset,
                base_power,
                data.improvement(target, name),
                OBJECTIVE_LABELS[name],
                selected,
                maximum="Maximum Improvement",
            )
            for name in (objective, other)
        ]
        if data.nprobes(target) * len(group) > 1:
            figures.append(
                probes_figure(
                    frequency,
                    offset,
                    data.probe_snr(target),
                    labels,
                    "SNR improvement [dB]",
                    selected,
                )
            )
    else:
        shared += f"<p>Maps are reported for qubit {group[0]}.</p>"

    if selected is not None and selected.candidate:
        i = offset.index(selected.offset)
        j = frequency.index(selected.frequency)
        failed = " and ".join(
            f"pump {edge}"
            for edge in edges(data.improvement(target, objective).shape, i, j)
        )
        report = shared + (
            f"<p><b>Fit failed</b>: the operating point lies at the edge of the scanned "
            f"{failed} range, thus the optimum may lie outside of it. Extend the scan "
            f"(e.g. changing the base pump power) and repeat it. The candidate point "
            f"is marked in the plot, and it is not applied.</p>"
        )
    elif fit is not None and selected is not None:
        columns = [
            "Pump Frequency [Hz]",
            "Pump Amplitude",
            "Pump power [dB]",
            OBJECTIVE_LABELS[fit.objective],
        ]
        values = [
            np.round(fit.frequency[target], 4),
            np.round(fit.offset[target], 4),
            np.round(fit.power[target], 4),
            np.round(fit.improvement[target], 4),
        ]
        if len(labels) != len(fit.probe_snr.get(target, [])):
            labels = [f"Probe {p}" for p in range(len(fit.probe_snr.get(target, [])))]
        for name, probe_values in (
            ("SNR improvement", fit.probe_snr),
            ("Gain", fit.probe_gain),
            ("Added noise", fit.probe_noise),
        ):
            columns += [f"{label} {name} [dB]" for label in labels]
            values += [np.round(v, 4) for v in probe_values.get(target, [])]
        report = shared + table_html(
            table_dict([target] * len(columns), columns, values)
        )
    else:
        report = shared
    return figures, report


def _update(results: TwpaSnrResults, platform: CalibrationPlatform, target: QubitId):
    """Set the pump to the optimal frequency and power.

    Candidate points are not results, thus there is nothing to apply. For the details
    of the update, cf. :func:`.utils.update_pump`.
    """
    if results.is_candidate(target):
        return
    update_pump(platform, target, results.frequency[target], results.power[target])


twpa_snr = Protocol(_acquisition, _fit, _plot, _update)
"""TWPA SNR improvement over the pump frequency and offset."""
