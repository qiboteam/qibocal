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

from ...auto.operation import Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import RangeLike, table_dict, table_html, to_range
from .utils import (
    Selected,
    TwpaData,
    TwpaProbeParameters,
    bracket_drift,
    bracket_magnitude,
    brackets,
    candidate_note,
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
    reference_notes,
    reference_shots,
    select,
    shared_note,
    shots,
    statistics,
    update_point,
    valid_reference,
)


@dataclass(kw_only=True)
class TwpaSnrParameters(TwpaProbeParameters):
    """Inputs of the TWPA SNR improvement scan."""

    amplitude: RangeLike
    """Range of amplitude (offset) values for TWPA sweep."""
    frequency: RangeLike
    """Range of TWPA frequency values for sweep."""
    max_values: int = 2_000_000
    """Maximum number of shots (I/Q pairs) of a target, held by a single execution.

    The pump offsets are split in batches accordingly, such that the acquisition
    memory of the controllers is not exceeded: the full scan needs ``nshots *
    noffsets * nfrequencies`` values for each target.
    """

    def __post_init__(self):
        super().__post_init__()
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
    reference_drift: dict[QubitId, list[float]] = field(default_factory=dict)
    """Change [dB] of the pump-off reference of each probe, across its sweep."""
    objective: SnrObjective = "snr"
    """Improvement optimized by the fit."""

    def __contains__(self, key: QubitId) -> bool:
        """Whether the fit produced a point for `key`."""
        return all(
            key in getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), dict)
            and not f.name.startswith("probe_")
            and f.name not in ("candidate", "reference_drift")
        )

    def is_candidate(self, qubit: QubitId) -> bool:
        """Whether the point of `qubit` is only a candidate."""
        return self.candidate.get(qubit, False)


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
    reference_value: dict[QubitId, list] = field(default_factory=dict)
    """Statistics of the shots with TWPA off, for each probe and bracket.

    The shape is ``(nprobes, nbrackets, 3)``, with the last axis as the one of
    :attr:`data`, and the brackets before and after the sweep of each probe, cf.
    :func:`.utils.brackets`.
    """

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.references(qubit).shape[0]

    def references(self, qubit: QubitId) -> npt.NDArray:
        """Statistics of the reference shots, ``(nprobes, nbrackets, 3)``."""
        return brackets(self.reference_value[qubit], 3)

    def _group(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Statistics with the pump on and off, of the probes on the pump of `qubit`.

        Returns:
            The statistics with the pump on, ``(noffsets, nfrequencies, nprobes, 3)``,
            and the magnitude and spread with the pump off, ``(nprobes,)``, averaged
            over the brackets (the spread in quadrature).
        """
        group = self.group(qubit)
        signal = np.concatenate([self[q] for q in group], axis=2)
        references = np.concatenate([self.references(q) for q in group])
        off = bracket_magnitude(references)
        spread = np.sqrt(np.mean(references[..., 2] ** 2, axis=1))
        valid_reference(off, qubit)
        if not np.all(np.isfinite(spread)) or np.any(np.isclose(spread, 0)):
            raise ValueError(
                f"TWPA-off reference for qubit {qubit} must have a finite and "
                "nonzero spread, for every probe."
            )
        return signal, off, spread

    def reference_drift(self, qubit: QubitId) -> npt.NDArray:
        """Drift [dB] of the pump-off reference of the probes on the pump of `qubit`."""
        return np.concatenate(
            [bracket_drift(self.references(q)) for q in self.group(qubit)]
        )

    def probe_gain(self, qubit: QubitId) -> npt.NDArray:
        """TWPA gain [dB] of each probe, with respect to its own reference.

        It is the ratio of the magnitudes of the mean signals. The probes are all the
        ones amplified by the pump of `qubit`, as listed by :meth:`group_probes`. The
        shape is ``(noffsets, nfrequencies, nprobes)``.
        """
        signal, off, _ = self._group(qubit)
        return 20 * np.log10(magnitude(signal[..., :2]) / off)

    def probe_noise(self, qubit: QubitId) -> npt.NDArray:
        """Noise [dB] added by the TWPA to each probe, with the shape of the gain.

        It is the ratio of the spreads of the shots, with and without pump, and it is
        the part of the gain which is not signal.
        """
        signal, _, spread = self._group(qubit)
        return 20 * np.log10(signal[..., 2] / spread)

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
    """Acquire the shots statistics over the pump, bracketed by TWPA-off references.

    As in :func:`.frequency_offset._acquisition`, the 2D hardware sweep over the pump
    frequency and offset of each probe is bracketed by references with the TWPA off.
    But the individual shots are acquired, to know their spread: they are reduced to
    their statistics as soon as downloaded. To bound the memory they need, the sweep is
    split in batches of pump offsets, according to ``params.max_values``.
    """
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes, params.detunings)

    freq_ranges = {
        q: to_range(params.frequency, center=pump.config[q].frequency) for q in targets
    }
    freq_sweeps = frequency_sweepers(pump, freq_ranges)
    frequencies = {q: s.values for q, s in zip(pump.unique.values(), freq_sweeps)}
    offsets = offset_sweepers(pump, to_range(params.amplitude))[0].values
    off = off_updates(platform, pump.unique, params.off_power)

    # the shots default to the platform ones, which the batches have to know
    nshots = shots(params, platform)
    nfrequencies = max(len(f) for f in frequencies.values())
    rows = max(1, params.max_values // (nshots * nfrequencies))

    def pump_off(probe) -> dict[QubitId, list[float]]:
        execution = reference(
            platform,
            sequence,
            handles,
            [probe],
            off,
            reference_shots(params, platform),
            params.relaxation_time,
            AveragingMode.SINGLESHOT,
        )[0]
        return {q: statistics(execution[q]).tolist() for q in targets}

    references: dict[QubitId, list] = {q: [] for q in targets}
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    for probe in updates:
        before = pump_off(probe)
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
        after = pump_off(probe)
        for qubit in targets:
            raw_data[qubit].append(np.concatenate(batches[qubit], axis=0))
            references[qubit].append([before[qubit], after[qubit]])

    return TwpaSnrData(
        data={q: np.stack(raw_data[q], axis=2) for q in targets},
        offset={q: offsets.tolist() for q in targets},
        frequency={
            q: frequencies[pump.unique[pump.channel[q]]].tolist() for q in targets
        },
        reference_value=references,
        probes=params.probes,
        detunings=params.detunings,
        power=pump.power,
        pump=pump.channel,
    )


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
    probe_snr, probe_gain, probe_noise, drift = {}, {}, {}, {}
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
        qubit_drift = data.reference_drift(qubit).tolist()
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
            drift[q] = qubit_drift
    return TwpaSnrResults(
        frequency=frequency,
        offset=offset,
        improvement=improvement,
        power=power,
        candidate=candidate,
        probe_snr=probe_snr,
        probe_gain=probe_gain,
        probe_noise=probe_noise,
        reference_drift=drift,
        objective=fitpars.objective,
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
    selected = Selected.of(fit, target)
    frequency, offset = data.frequency[target], data.offset[target]
    base_power = data.power.get(target, 0.0)
    labels = probe_labels(data, target)

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

    if fit is None or selected is None:
        return figures, shared
    if len(labels) != len(fit.probe_snr.get(target, [])):
        # results not matching the probes grouping of the data
        labels = [f"Probe {p}" for p in range(len(fit.probe_snr.get(target, [])))]
    # the reference does not depend on the point, thus it is checked even for candidates
    notes = reference_notes(
        labels, fit.reference_drift.get(target, []), fit.probe_gain.get(target, [])
    )
    if selected.candidate:
        return figures, shared + notes + candidate_note(
            data.improvement(target, objective).shape, offset, frequency, selected
        )
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
    for name, probe_values in (
        ("SNR improvement", fit.probe_snr),
        ("Gain", fit.probe_gain),
        ("Added noise", fit.probe_noise),
        ("reference drift", fit.reference_drift),
    ):
        stat = probe_values.get(target, [])
        columns += [f"{label} {name} [dB]" for label in labels[: len(stat)]]
        values += [np.round(v, 4) for v in stat]
    report = (
        shared
        + notes
        + table_html(table_dict([target] * len(columns), columns, values))
    )
    return figures, report


twpa_snr = Protocol(_acquisition, _fit, _plot, update_point)
"""TWPA SNR improvement over the pump frequency and offset."""
