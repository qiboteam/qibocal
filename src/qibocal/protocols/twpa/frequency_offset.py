"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

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
    update_point,
    valid_reference,
)


@dataclass(kw_only=True)
class TwpaFrequencyOffsetParameters(TwpaProbeParameters):
    amplitude: RangeLike
    """Range of amplitude (offset) values for TWPA sweep.

    Including the null offset enables the in-sweep reference, cf.
    :data:`ReferenceKind`.
    """
    frequency: RangeLike
    """Range of TWPA frequency values for sweep."""


GainObjective = Literal["minimum", "average"]
"""Aggregation of the probes' gains optimized by the fit.

- ``minimum``: the minimum gain across the probes, i.e. the gain of the worst probe.
- ``average``: the ratio of the probe-averaged magnitudes, with and without pump,
  dominated by the probes with the largest signal.
"""

ReferenceKind = Literal["pump_off", "zero_offset"]
"""Pump-off reference normalizing the signal, into the gain.

- ``pump_off``: the pump turned off by lowering its power, acquired before and after
  the sweep of each probe, and averaged.
- ``zero_offset``: the signal at the null pump offset, within the sweep, averaged over
  the pump frequencies. It is interleaved with the whole scan by the averaging, thus
  it follows the same drifts as the signal, but it is subject to the LO leakage of the
  pump. It requires the null offset to be part of the scanned ones.

Their ratio is always reported, as the gain of the unpumped TWPA, which should vanish.
"""


@dataclass
class TwpaFrequencyOffsetFitParameters:
    """Fit parameters of the TWPA operating point selection."""

    neighbourhood: int = 1
    """Half-width of the neighbourhood of each grid point, in grid steps.

    The operating point maximizes the minimum gain over the neighbourhood, such that
    it lies within the gain plateau, away from the over-pumping region, where small
    drifts of the pump cause large gain losses. ``0`` selects the plain maximum.
    """
    objective: GainObjective = "minimum"
    """Gain optimized across the probes, cf. :data:`GainObjective`.

    With a single probe, the two coincide.
    """
    reference: ReferenceKind = "pump_off"
    """Reference of the gain, cf. :data:`ReferenceKind`."""

    def __post_init__(self):
        if self.objective not in get_args(GainObjective):
            raise ValueError(
                f"Unknown gain objective '{self.objective}', expected one of "
                f"{get_args(GainObjective)}."
            )
        if self.reference not in get_args(ReferenceKind):
            raise ValueError(
                f"Unknown reference '{self.reference}', expected one of "
                f"{get_args(ReferenceKind)}."
            )


OPTIONAL_RESULTS = ("candidate", "probe_gain", "reference_drift", "zero_offset_error")
"""Results fields missing for some targets, or in results of previous versions."""


@dataclass
class TwpaFrequencyOffsetResults(Results):
    frequency: dict[QubitId, float]
    """Pump frequency [Hz]."""
    offset: dict[QubitId, float]
    """Pump offset [a.u.]."""
    gain: dict[QubitId, float]
    """TWPA gain [dB], aggregated across the probes as optimized by the fit."""
    power: dict[QubitId, float] = field(default_factory=dict)
    """Pump power [dB], corresponding to the optimal offset."""
    candidate: dict[QubitId, bool] = field(default_factory=dict)
    """Whether the point is only a candidate, rather than the result.

    It is a candidate when it lies at the edge of the scanned ranges, thus the optimum
    may lie outside of them. Candidates are reported, but not applied.
    """
    probe_gain: dict[QubitId, list[float]] = field(default_factory=dict)
    """TWPA gain [dB] of each probe at the selected point, cf.
    :meth:`TwpaFrequencyOffsetData.probe_gain`."""
    reference_drift: dict[QubitId, list[float]] = field(default_factory=dict)
    """Change [dB] of the pump-off reference of each probe, across its sweep."""
    zero_offset_error: dict[QubitId, list[float]] = field(default_factory=dict)
    """Gain [dB] of each probe at the null offset, against the pump-off reference.

    It should vanish: it measures the error of the pump-off reference, with respect
    to the in-sweep one. Only available if the null offset is scanned.
    """
    objective: GainObjective = "average"
    """Gain optimized by the fit, cf. :attr:`TwpaFrequencyOffsetFitParameters.objective`.

    The default is the one of the results of previous versions.
    """
    reference: ReferenceKind = "pump_off"
    """Reference of the gain, cf. :attr:`TwpaFrequencyOffsetFitParameters.reference`."""

    def __contains__(self, key: QubitId) -> bool:
        """Whether the fit produced a point for `key`.

        Overwritten since some fields are optional, cf. :data:`OPTIONAL_RESULTS`.
        """
        return all(
            key in getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), dict)
            and f.name not in OPTIONAL_RESULTS
        )

    def is_candidate(self, qubit: QubitId) -> bool:
        """Whether the point of `qubit` is only a candidate."""
        return self.candidate.get(qubit, False)


@dataclass
class TwpaFrequencyOffsetData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired, with shape ``(noffsets, nfrequencies, nprobes, 2)``."""
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept."""
    reference_value: dict[QubitId, list] = field(default_factory=dict)
    """Pump-off reference I/Q points, with shape ``(nprobes, nbrackets, 2)``.

    The brackets are the references before and after the sweep of each probe, cf.
    :func:`.utils.brackets`.
    """

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.references(qubit).shape[0]

    def references(self, qubit: QubitId) -> npt.NDArray:
        """Pump-off references, with shape ``(nprobes, nbrackets, 2)``."""
        return brackets(self.reference_value[qubit], 2)

    def zero_offset(self, qubit: QubitId) -> int | None:
        """Index of the null pump offset, if scanned."""
        null = np.flatnonzero(np.isclose(self.offset[qubit], 0))
        return int(null[0]) if len(null) > 0 else None

    def _group_magnitudes(
        self, qubit: QubitId, reference: ReferenceKind = "pump_off"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Pump on and off magnitudes of all the probes on the pump of `qubit`.

        The shapes are ``(noffsets, nfrequencies, nprobes)`` and ``(nprobes,)``.
        """
        group = self.group(qubit)
        signal = np.concatenate([magnitude(self[q]) for q in group], axis=2)
        if reference == "zero_offset":
            zero = self.zero_offset(qubit)
            if zero is None:
                raise ValueError(
                    f"The in-sweep reference of qubit {qubit} requires the null pump "
                    "offset to be scanned."
                )
            off = np.mean(signal[zero], axis=0)
        else:
            off = np.concatenate([bracket_magnitude(self.references(q)) for q in group])
        valid_reference(off, qubit)
        return signal, off

    def reference_drift(self, qubit: QubitId) -> npt.NDArray:
        """Drift [dB] of the pump-off reference of the probes on the pump of `qubit`."""
        return np.concatenate(
            [bracket_drift(self.references(q)) for q in self.group(qubit)]
        )

    def zero_offset_error(self, qubit: QubitId) -> npt.NDArray | None:
        """Gain [dB] at the null offset against the pump-off reference, per probe.

        It is the error of the pump-off reference, which should vanish. ``None`` if the
        null offset is not scanned.
        """
        if self.zero_offset(qubit) is None:
            return None
        _, zero = self._group_magnitudes(qubit, "zero_offset")
        _, off = self._group_magnitudes(qubit, "pump_off")
        return 20 * np.log10(zero / off)

    def averaged_gain(
        self, qubit: QubitId, reference: ReferenceKind = "pump_off"
    ) -> npt.NDArray:
        """Ratio of the probe-averaged magnitudes [dB], with and without pump.

        The average runs over all the probes amplified by the pump of `qubit`, cf.
        :meth:`group_probes`.
        """
        signal, off = self._group_magnitudes(qubit, reference)
        return 20 * np.log10(np.mean(signal, axis=2) / np.mean(off))

    def probe_gain(
        self, qubit: QubitId, reference: ReferenceKind = "pump_off"
    ) -> npt.NDArray:
        """TWPA gain [dB] of each probe, with respect to its own reference.

        Unlike :meth:`averaged_gain`, it is not dominated by the probes with the
        largest signal. The probes are all the ones amplified by the pump of `qubit`,
        i.e. those of all the targets sharing it, as listed by :meth:`group_probes`.
        The shape is ``(noffsets, nfrequencies, nprobes)``.
        """
        signal, off = self._group_magnitudes(qubit, reference)
        return 20 * np.log10(signal / off)

    def gain(
        self,
        qubit: QubitId,
        objective: GainObjective,
        reference: ReferenceKind = "pump_off",
    ) -> npt.NDArray:
        """TWPA gain [dB] map, aggregated across the probes as in `objective`.

        ``minimum`` is the gain of the worst probe, which limits a multiplexed readout.
        """
        if objective == "minimum":
            return np.min(self.probe_gain(qubit, reference), axis=2)
        return self.averaged_gain(qubit, reference)


def _acquisition(
    params: TwpaFrequencyOffsetParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaFrequencyOffsetData:
    """Acquire the 2D gain landscape, bracketed by pump-off references, for each probe.

    For each probe, acquire a reference with the TWPA off, then sweep the TWPA
    amplitude (offset) and frequency concurrently using a 2D sweeper, and acquire a
    second reference. The two references bracket the sweep: their average follows a
    linear drift, and their difference tells whether the reference is stable.

    Note on the probe frequencies handling:
    The TWPA gain oscillates quite fast in the probe frequency. Instead of obtaining
    the best gain on average across an arbitrary continuous frequency range, we optimize
    specifically for the frequencies of the expected readout resonators, possibly
    detuned (cf. :attr:`.utils.TwpaProbeParameters.detunings`). When `params.probes` and
    `params.detunings` are omitted, the protocol performs a single acquisition run
    where each target is probed exclusively at its own calibrated readout frequency.
    Otherwise, the protocol iterates over each probe in software. Because not all
    control electronics support sweeping an arbitrary on-board list of values, this
    multi-probe evaluation is implemented as a software loop. The primary speed-up is
    achieved through the on-board 2D hardware sweep over the TWPA pump signal
    parameters (amplitude and frequency).

    Only the averages are acquired, which are all the gain needs: it is the cheapest TWPA
    scan, but it cannot tell the noise added by the TWPA, cf. :mod:`.snr`.
    """
    # The sequence is purely made by simultaneous rectangular readouts, whose duration
    # and amplitude are given as inputs
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes, params.detunings)
    freq_ranges = {
        q: to_range(params.frequency, center=pump.config[q].frequency) for q in targets
    }
    freq_sweeps = frequency_sweepers(pump, freq_ranges)
    offset_sweeps = offset_sweepers(pump, to_range(params.amplitude))
    off = off_updates(platform, pump.unique, params.off_power)

    def pump_off(probe) -> dict[QubitId, npt.NDArray]:
        return reference(
            platform,
            sequence,
            handles,
            [probe],
            off,
            reference_shots(params, platform),
            params.relaxation_time,
            AveragingMode.CYCLIC,
        )[0]

    references: dict[QubitId, list] = {q: [] for q in targets}
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    for probe in updates:
        before = pump_off(probe)
        results = platform.execute(
            [sequence],
            [offset_sweeps, freq_sweeps],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=[probe],
        )
        after = pump_off(probe)
        for qubit in targets:
            raw_data[qubit].append(results[handles[qubit]])
            references[qubit].append([before[qubit].tolist(), after[qubit].tolist()])

    return TwpaFrequencyOffsetData(
        data={q: np.stack(raw_data[q], axis=2) for q in targets},
        offset={q: offset_sweeps[0].values.tolist() for q in targets},
        frequency={q: np.arange(*freq_ranges[q]).tolist() for q in targets},
        reference_value=references,
        probes=params.probes,
        detunings=params.detunings,
        power=pump.power,
        pump=pump.channel,
    )


def _fit(
    data: TwpaFrequencyOffsetData,
    fitpars: TwpaFrequencyOffsetFitParameters | None = None,
) -> TwpaFrequencyOffsetResults:
    """Select a robust operating point, within the gain plateau.

    After aggregating the gain across the evaluated probes, as in the `objective` fit
    parameter (by default, their minimum), select the TWPA frequency and offset as in
    :func:`.utils.select`. The maximum gain may be just below the over-pumping region,
    where the gain drops sharply, and small drifts of the pump cause large gain losses:
    maximizing the gain over a neighbourhood trades a fraction of the gain for
    stability.

    The point does not depend on the reference, which only shifts the gain map of each
    probe by a constant. Its gain does, thus the stability of the reference is reported
    along with it.

    Targets sharing a pump are amplified by the same working point: the gain is
    aggregated over all their probes, and the same point is selected for all of them.
    """
    fitpars = TwpaFrequencyOffsetFitParameters() if fitpars is None else fitpars
    frequency, offset, gain, power, candidate = {}, {}, {}, {}, {}
    probe_gain, drift, zero_error = {}, {}, {}
    for qubit in data.qubits:
        if qubit in frequency:
            # already selected, together with a target sharing its pump
            continue
        gain_map = data.gain(qubit, fitpars.objective, fitpars.reference)
        i, j, is_candidate = select(gain_map, fitpars.neighbourhood)
        point_gain = data.probe_gain(qubit, fitpars.reference)[i, j].tolist()
        qubit_drift = data.reference_drift(qubit).tolist()
        qubit_error = data.zero_offset_error(qubit)
        for q in data.group(qubit):
            candidate[q] = is_candidate
            frequency[q] = data.frequency[q][j]
            offset[q] = data.offset[q][i]
            gain[q] = gain_map[i, j]
            power[q] = data.power.get(q, 0.0) + 20 * np.log10(abs(offset[q]))
            probe_gain[q] = point_gain
            drift[q] = qubit_drift
            if qubit_error is not None:
                zero_error[q] = qubit_error.tolist()
    return TwpaFrequencyOffsetResults(
        frequency=frequency,
        offset=offset,
        gain=gain,
        power=power,
        candidate=candidate,
        probe_gain=probe_gain,
        reference_drift=drift,
        zero_offset_error=zero_error,
        objective=fitpars.objective,
        reference=fitpars.reference,
    )


OBJECTIVE_LABELS = {
    "minimum": "Minimum TWPA Gain across probes [dB]",
    "average": "Average TWPA Gain [dB]",
}
"""Labels of the gain maps aggregated across the probes."""


def _plot(
    data: TwpaFrequencyOffsetData,
    fit: TwpaFrequencyOffsetResults | None,
    target: QubitId,
):
    """Plot the TWPA gain, and report fit results - if any.

    The first figure displays the gain map optimized by the fit (by default, the
    minimum gain across the probes) as a 2D heatmap versus pump frequency (horizontal
    axis) and pump amplitude/offset (vertical axis). If fit results are available, the
    selected point is highlighted with a marker. With several probes, the other
    aggregated map (the average gain, by default) and the maps of each probe follow,
    since the aggregated maps alone hide how the gain is distributed among the probes.

    The report lists the gain of each probe at the point, the drift of its pump-off
    reference across the sweep and, if the null offset is scanned, the error of the
    pump-off reference with respect to the in-sweep one, with warnings if the
    reference is not reliable.

    The probes are all the ones amplified by the pump of `target`, including those of
    the other targets sharing it. Since the maps are the same for all of them, they
    are only plotted for the first one, and the others refer to it.
    """
    objective = fit.objective if fit is not None else "minimum"
    ref = fit.reference if fit is not None else "pump_off"
    others = [other for other in OBJECTIVE_LABELS if other != objective]
    group = data.group(target)
    shared = shared_note(data, target)
    selected = Selected.of(fit, target)
    frequency, offset = data.frequency[target], data.offset[target]
    base_power = data.power.get(target, 0.0)
    labels = probe_labels(data, target)

    figures = []
    if target == group[0]:
        maps = [objective] + (others if len(labels) > 1 else [])
        figures += [
            map_figure(
                frequency,
                offset,
                base_power,
                data.gain(target, name, ref),
                OBJECTIVE_LABELS[name],
                selected,
            )
            for name in maps
        ]
        if len(labels) > 1:
            figures.append(
                probes_figure(
                    frequency,
                    offset,
                    data.probe_gain(target, ref),
                    labels,
                    "TWPA Gain [dB]",
                    selected,
                )
            )
    else:
        shared += f"<p>Gain maps are reported for qubit {group[0]}.</p>"

    if fit is None or selected is None:
        return figures, shared
    # the reference does not depend on the point, thus it is checked even for candidates
    probe_gain = fit.probe_gain.get(target, [])
    if len(labels) != len(probe_gain):
        # results not matching the probes grouping of the data
        labels = [f"Probe {p}" for p in range(len(probe_gain))]
    notes = reference_notes(labels, fit.reference_drift.get(target, []), probe_gain)
    columns, values = [], []
    for name, stat in (
        ("reference drift", fit.reference_drift.get(target, [])),
        ("zero offset error", fit.zero_offset_error.get(target, [])),
    ):
        columns += [f"{label} {name} [dB]" for label in labels[: len(stat)]]
        values += [np.round(value, 4) for value in stat]
    if selected.candidate:
        note = candidate_note(
            data.gain(target, objective, ref).shape, offset, frequency, selected
        )
    else:
        note = ""
        point = [
            ("Pump Frequency [Hz]", fit.frequency[target]),
            ("Pump Amplitude", fit.offset[target]),
            ("Pump power [dB]", fit.power[target]),
            (OBJECTIVE_LABELS[fit.objective], fit.gain[target]),
        ]
        if len(labels) > 1:
            point += [(f"{label} Gain [dB]", g) for label, g in zip(labels, probe_gain)]
        columns = [name for name, _ in point] + columns
        values = [np.round(value, 4) for _, value in point] + values
    table = (
        table_html(table_dict([target] * len(columns), columns, values))
        if columns
        else ""
    )
    return figures, shared + notes + note + table


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot, update_point)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object."""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
