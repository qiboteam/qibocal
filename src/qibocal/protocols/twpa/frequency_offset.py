"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

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
class TwpaFrequencyOffsetParameters(Parameters):
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
    """Pump power [dB] of the TWPA-off reference.

    It overrides :data:`PUMP_OFF_POWER`, for instruments supporting different power
    ranges. It should be low enough to leave the TWPA unpumped, and to suppress the
    LO leakage.
    """


GainObjective = Literal["minimum", "average"]
"""Aggregation of the probes' gains optimized by the fit.

- ``minimum``: the minimum gain across the probes, i.e. the gain of the worst probe.
- ``average``: the ratio of the probe-averaged magnitudes, with and without pump,
  dominated by the probes with the largest signal.
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

    def __post_init__(self):
        if self.objective not in get_args(GainObjective):
            raise ValueError(
                f"Unknown gain objective '{self.objective}', expected one of "
                f"{get_args(GainObjective)}."
            )


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
    objective: GainObjective = "average"
    """Gain optimized by the fit, cf. :attr:`TwpaFrequencyOffsetFitParameters.objective`.

    The default is the one of the results of previous versions.
    """

    def __contains__(self, key: QubitId) -> bool:
        """Whether the fit produced a point for `key`.

        Overwritten since `candidate` and `probe_gain` are missing in results of
        previous versions.
        """
        return all(
            key in getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), dict)
            and f.name not in ("candidate", "probe_gain")
        )

    def is_candidate(self, qubit: QubitId) -> bool:
        """Whether the point of `qubit` is only a candidate."""
        return self.candidate.get(qubit, False)


@dataclass
class TwpaFrequencyOffsetData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept."""
    reference_value: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Reference values with TWPA off for each probe frequency.

    Reference measurements are I/Q points themselves, as the acquired data.
    """

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.reference_value_array(qubit).shape[0]

    def _group_magnitudes(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Pump on and off magnitudes of all the probes on the pump of `qubit`.

        The shapes are ``(noffsets, nfrequencies, nprobes)`` and ``(nprobes,)``.
        """
        group = self.group(qubit)
        signal = np.concatenate([magnitude(self[q]) for q in group], axis=2)
        reference = np.concatenate(
            [magnitude(self.reference_value_array(q)) for q in group]
        )
        return signal, reference

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray:
        """Return reference value as a numpy array.

        The shape is ``(nprobes, 2)``, with ``nprobes`` the number of probes used, and 2
        for the I/Q measurements.
        """
        return np.array(self.reference_value[qubit]).reshape(-1, 2)

    def averaged_gain(self, qubit: QubitId) -> npt.NDArray:
        """Ratio of the probe-averaged magnitudes [dB], with and without pump.

        The average runs over all the probes amplified by the pump of `qubit`, cf.
        :meth:`group_probes`.
        """
        signal, reference = self._group_magnitudes(qubit)
        reference = np.mean(reference)
        if not np.isfinite(reference) or np.isclose(reference, 0):
            raise ValueError(
                f"TWPA-off reference magnitude for qubit {qubit} must be finite "
                "and nonzero."
            )
        return 20 * np.log10(np.mean(signal, axis=2) / reference)

    def probe_gain(self, qubit: QubitId) -> npt.NDArray:
        """TWPA gain [dB] of each probe, with respect to its own reference.

        Unlike :meth:`averaged_gain`, it is not dominated by the probes with the
        largest signal. The probes are all the ones amplified by the pump of `qubit`,
        i.e. those of all the targets sharing it, as listed by :meth:`group_probes`.
        The shape is ``(noffsets, nfrequencies, nprobes)``.
        """
        signal, reference = self._group_magnitudes(qubit)
        if not np.all(np.isfinite(reference)) or np.any(np.isclose(reference, 0)):
            raise ValueError(
                f"TWPA-off reference magnitude for qubit {qubit} must be finite "
                "and nonzero, for every probe."
            )
        return 20 * np.log10(signal / reference)

    def minimum_gain(self, qubit: QubitId) -> npt.NDArray:
        """Minimum TWPA gain [dB] across the probes, cf. :meth:`probe_gain`.

        It is the gain of the worst probe, which limits a multiplexed readout.
        """
        return np.min(self.probe_gain(qubit), axis=2)

    def gain(self, qubit: QubitId, objective: "GainObjective") -> npt.NDArray:
        """TWPA gain [dB] map, aggregated across the probes as in `objective`."""
        return (
            self.minimum_gain(qubit)
            if objective == "minimum"
            else self.averaged_gain(qubit)
        )


def _acquisition(
    params: TwpaFrequencyOffsetParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaFrequencyOffsetData:
    """Acquire reference scan and 2D gain landscape, for each probe.

    First perform a scan over the readout probe with the TWPA off, then sweep the
    TWPA amplitude (offset) and frequency concurrently using a 2D sweeper.

    Note on the probe frequencies handling:
    The TWPA gain oscillates quite fast in the probe frequency. Instead of obtaining
    the best gain on average across an arbitrary continuous frequency range, we optimize
    specifically for the frequencies of the expected readout resonators.
    When `params.probes` is omitted (None), the protocol performs a single acquisition run
    where each target qubit is updated and probed exclusively at its own calibrated
    readout frequency.
    When `params.probes` is provided as a discrete list of frequencies, the protocol
    iterates over each probe frequency in software, applying it to all targets across
    successive runs. Because not all control electronics support sweeping an arbitrary
    on-board list of values, this multi-probe evaluation is implemented as a software loop.
    The primary speed-up is achieved through the on-board 2D hardware sweep over
    the TWPA pump signal parameters (amplitude and frequency).

    Only the averages are acquired, which are all the gain needs: it is the cheapest TWPA
    scan, but it cannot tell the noise added by the TWPA, cf. :mod:`.snr`.
    """
    # The sequence is purely made by simultaneous rectangular readouts, whose duration
    # and amplitude are given as inputs
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes)

    # TWPA frequency ranges
    freq_ranges = {
        q: to_range(params.frequency, center=pump.config[q].frequency) for q in targets
    }

    # TWPA amplitude (offset) range (linear sweep)
    offset_range = to_range(params.amplitude)

    # Build 2D sweepers over TWPA pump parameters.
    # Use Sweeper-generated values to ensure consistency with backend semantics.
    freq_sweeps = frequency_sweepers(pump, freq_ranges)
    offset_sweeps = offset_sweepers(pump, offset_range)
    offset_values = offset_sweeps[0].values

    # 1. Reference measurements with TWPA off
    # The pump is turned off by lowering its power, which also suppresses the LO
    # leakage, unlike a null offset.
    references = reference(
        platform,
        sequence,
        handles,
        updates,
        off_updates(platform, pump.unique, params.off_power),
        params.nshots,
        params.relaxation_time,
        AveragingMode.CYCLIC,
    )
    reference_data: dict[QubitId, list[list[float]]] = {
        q: [probe[q].tolist() for probe in references] for q in targets
    }

    # 2. 2D TWPA sweeps (amplitude and frequency)
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    for probe_updates_ in updates:
        results = platform.execute(
            [sequence],
            [offset_sweeps, freq_sweeps],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=[probe_updates_],
        )
        for qubit in targets:
            raw_data[qubit].append(results[handles[qubit]])

    data = TwpaFrequencyOffsetData(
        offset={q: offset_values.tolist() for q in targets},
        frequency={q: np.arange(*freq_ranges[q]).tolist() for q in targets},
        reference_value=reference_data,
        probes=params.probes,
        power=pump.power,
        pump=pump.channel,
    )
    for qubit in targets:
        data.data[qubit] = np.stack(raw_data[qubit], axis=2)

    return data


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

    Targets sharing a pump are amplified by the same working point: the gain is
    aggregated over all their probes, and the same point is selected for all of them.
    """
    fitpars = TwpaFrequencyOffsetFitParameters() if fitpars is None else fitpars
    frequency = {}
    offset = {}
    gain = {}
    power = {}
    candidate = {}
    probe_gain = {}
    for qubit in data.qubits:
        if qubit in frequency:
            # already selected, together with a target sharing its pump
            continue
        gain_map = data.gain(qubit, fitpars.objective)
        i, j, is_candidate = select(gain_map, fitpars.neighbourhood)
        point_gain = data.probe_gain(qubit)[i, j].tolist()
        for q in data.group(qubit):
            candidate[q] = is_candidate
            frequency[q] = data.frequency[q][j]
            offset[q] = data.offset[q][i]
            gain[q] = gain_map[i, j]
            power[q] = data.power.get(q, 0.0) + 20 * np.log10(abs(offset[q]))
            probe_gain[q] = point_gain
    return TwpaFrequencyOffsetResults(
        frequency=frequency,
        offset=offset,
        gain=gain,
        power=power,
        candidate=candidate,
        probe_gain=probe_gain,
        objective=fitpars.objective,
    )


OBJECTIVE_LABELS = {
    "minimum": "Minimum TWPA Gain across probes [dB]",
    "average": "Average TWPA Gain [dB]",
}
"""Labels of the gain maps aggregated across the probes."""


def _selected(
    fit: TwpaFrequencyOffsetResults | None, target: QubitId
) -> Selected | None:
    """The point selected by the fit for `target`, if any."""
    if fit is None or target not in fit:
        return None
    return Selected(
        frequency=fit.frequency[target],
        offset=fit.offset[target],
        candidate=fit.is_candidate(target),
    )


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

    The probes are all the ones amplified by the pump of `target`, including those of
    the other targets sharing it. Since the maps are the same for all of them, they
    are only plotted for the first one, and the others refer to it.
    """
    objective = fit.objective if fit is not None else "minimum"
    others = [other for other in OBJECTIVE_LABELS if other != objective]
    group = data.group(target)
    shared = shared_note(data, target)
    selected = _selected(fit, target)
    frequency, offset = data.frequency[target], data.offset[target]
    base_power = data.power.get(target, 0.0)
    labels = probe_labels(data.probes, group, data.group_probes(target))

    figures = []
    if target == group[0]:
        figures.append(
            map_figure(
                frequency,
                offset,
                base_power,
                data.gain(target, objective),
                OBJECTIVE_LABELS[objective],
                selected,
            )
        )
    else:
        shared += f"<p>Gain maps are reported for qubit {group[0]}.</p>"
    nprobes = data.probe_gain(target).shape[2]
    if nprobes > 1 and target == group[0]:
        figures += [
            map_figure(
                frequency,
                offset,
                base_power,
                data.gain(target, other),
                OBJECTIVE_LABELS[other],
                selected,
            )
            for other in others
        ]
        figures.append(
            probes_figure(
                frequency,
                offset,
                data.probe_gain(target),
                labels,
                "TWPA Gain [dB]",
                selected,
            )
        )

    if selected is not None and selected.candidate:
        i = data.offset[target].index(selected.offset)
        j = data.frequency[target].index(selected.frequency)
        failed = " and ".join(
            f"pump {edge}" for edge in edges(data.gain(target, objective).shape, i, j)
        )
        fitting_report = shared + (
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
            np.round(fit.gain[target], 4),
        ]
        probe_gain = fit.probe_gain.get(target, [])
        if len(probe_gain) > 1:
            if len(labels) != len(probe_gain):
                # results not matching the probes grouping of the data
                labels = [f"Probe {p}" for p in range(len(probe_gain))]
            columns += [f"{label} Gain [dB]" for label in labels]
            values += [np.round(value, 4) for value in probe_gain]
        fitting_report = shared + table_html(
            table_dict([target] * len(columns), columns, values)
        )
    else:
        fitting_report = shared

    return figures, fitting_report


def _update(
    results: TwpaFrequencyOffsetResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    """Set the pump to the optimal frequency and power.

    Targets sharing the same pump are assigned the same point by the fit, thus their
    updates coincide. For the details of the update, cf. :func:`.utils.update_pump`.

    Candidate points are not results, thus there is nothing to apply.
    """
    if results.is_candidate(target):
        return
    update_pump(platform, target, results.frequency[target], results.power[target])


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot, _update)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object."""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
