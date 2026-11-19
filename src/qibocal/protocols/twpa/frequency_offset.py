"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

from dataclasses import dataclass, field, fields
from typing import Literal, cast, get_args

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    Acquisition,
    AcquisitionType,
    AveragingMode,
    ChannelId,
    IqConfig,
    OscillatorConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)
from scipy.constants import nano
from scipy.ndimage import minimum_filter

from ...auto.operation import Data, Parameters, Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import (
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
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
class TwpaFrequencyOffsetData(Data):
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
    probes: list[float] | None = None
    """List of probe frequencies evaluated.

    ``None`` if native readout frequencies were used.
    """
    power: dict[QubitId, float] = field(default_factory=dict)
    """Configured base pump power [dB] for each target."""
    pump: dict[QubitId, str] = field(default_factory=dict)
    """TWPA pump channel amplifying the readout of each target.

    Missing in data of previous versions, whose targets are then treated as
    amplified by distinct pumps.
    """

    def group(self, qubit: QubitId) -> list[QubitId]:
        """Targets amplified by the same pump as `qubit`, in acquisition order.

        Their probes are all amplified by the same pump working point, thus the gain
        is aggregated over all of them, and a single point is selected for the pump.
        """
        pump = self.pump.get(qubit)
        if pump is None:
            return [qubit]
        return [q for q in self.qubits if self.pump.get(q) == pump]

    def group_probes(self, qubit: QubitId) -> list[tuple[QubitId, int]]:
        """Probes amplified by the pump of `qubit`, as ``(target, probe index)``.

        They are ordered as the last axis of :meth:`probe_gain`.
        """
        return [
            (q, p)
            for q in self.group(qubit)
            for p in range(self.reference_value_array(q).shape[0])
        ]

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


PUMP_OFF_POWER = -60.0
"""Pump power [dB] of the TWPA-off reference.

Low enough to leave the TWPA unpumped, while within the range of both standalone
oscillators and controllers' ports, in the instruments it has been tuned on. It can
be overridden through :attr:`TwpaFrequencyOffsetParameters.off_power`.
"""


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
    """
    acquisition: dict[QubitId, ChannelId] = {}
    for q in targets:
        acq = platform.qubits[q].acquisition
        if acq is None:
            raise ValueError(f"Acquisition channel for qubit {q} not defined.")
        acquisition[q] = acq

    # The sequence is purely made by simultaneous rectangular readouts, whose duration
    # and amplitude are given as inputs
    readouts = {
        q: Readout(
            probe=Pulse(
                amplitude=params.probe_amplitude,
                duration=params.probe_duration,
                envelope=Rectangular(),
            ),
            acquisition=Acquisition(duration=params.probe_duration),
        )
        for q in targets
    }
    sequence = PulseSequence([(acquisition[q], readouts[q]) for q in targets])
    acquisition_handles = {q: readouts[q].acquisition.id for q in targets}

    twpa_channels = {}
    twpa_configs = {}
    twpa_powers = {}
    for qubit in targets:
        pump = platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        if pump is None:
            raise ValueError(
                f"Qubit {qubit} does not have a TWPA pump channel configured."
            )
        twpa_channels[qubit] = pump
        twpa_configs[qubit] = cast(OscillatorConfig | IqConfig, platform.config(pump))
        twpa_powers[qubit] = _pump_power(platform, pump)

    # Deduplicate TWPA channels preserving association to a target qubit
    unique_twpa_channels: dict[str, QubitId] = {}
    for q in targets:
        ch = twpa_channels[q]
        if ch not in unique_twpa_channels:
            unique_twpa_channels[ch] = q

    # Probe configurations to evaluate
    if params.probes is not None:
        probe_updates_list = [
            {platform.qubits[q].probe: {"frequency": probe} for q in targets}
            for probe in params.probes
        ]
    else:
        probe_updates_list = [
            {
                platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
                for q in targets
            }
        ]

    # TWPA frequency ranges
    freq_ranges = {
        q: to_range(params.frequency, center=twpa_configs[q].frequency) for q in targets
    }

    # TWPA amplitude (offset) range (linear sweep)
    offset_range = to_range(params.amplitude)

    # Build 2D sweepers over TWPA pump parameters.
    # Use Sweeper-generated values to ensure consistency with backend semantics.
    freq_sweeps = [
        Sweeper(
            parameter=Parameter.frequency,
            range=freq_ranges[q],
            channels=[ch],
        )
        for ch, q in unique_twpa_channels.items()
    ]
    offset_sweeps = [
        Sweeper(
            parameter=Parameter.offset,
            range=offset_range,
            channels=[ch],
        )
        for ch in unique_twpa_channels
    ]
    offset_values = offset_sweeps[0].values
    if np.any(np.abs(offset_values) > 1.0):
        raise ValueError("TWPA amplitude values must be between -1 and 1.")

    # Reference value acquisition (TWPA off)
    reference_data: dict[QubitId, list[list[float]]] = {q: [] for q in targets}
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}

    # 1. Reference measurements with TWPA off
    # The pump is turned off by lowering its power, which also suppresses the LO
    # leakage, unlike a null offset.
    off_power = PUMP_OFF_POWER if params.off_power is None else params.off_power
    off_updates = {
        _power_config(platform, ch): {"power": off_power} for ch in unique_twpa_channels
    }
    for probe_updates in probe_updates_list:
        ref_results = platform.execute(
            [sequence],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=[probe_updates, off_updates],
        )
        for qubit in targets:
            reference_data[qubit].append(
                ref_results[acquisition_handles[qubit]].tolist()
            )

    # 2. 2D TWPA sweeps (amplitude and frequency)
    for probe_updates in probe_updates_list:
        updates = [probe_updates]
        results = platform.execute(
            [sequence],
            [offset_sweeps, freq_sweeps],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )
        for qubit in targets:
            raw_data[qubit].append(results[acquisition_handles[qubit]])

    data = TwpaFrequencyOffsetData(
        offset={q: offset_values.tolist() for q in targets},
        frequency={q: np.arange(*freq_ranges[q]).tolist() for q in targets},
        reference_value=reference_data,
        probes=params.probes,
        power=twpa_powers,
        pump=twpa_channels,
    )
    for qubit in targets:
        data.data[qubit] = np.stack(raw_data[qubit], axis=2)

    return data


def _power_config(platform: CalibrationPlatform, pump: ChannelId) -> str:
    """Configuration holding the pump power.

    The pump is either a standalone oscillator, whose configuration holds the power,
    or an IQ channel of a controller, whose power is set by its LO.
    """
    if isinstance(platform.config(pump), OscillatorConfig):
        return pump
    return _pump_lo(platform, pump)


def _pump_power(platform: CalibrationPlatform, pump: ChannelId) -> float:
    """Configured pump power."""
    return cast(OscillatorConfig, platform.config(_power_config(platform, pump))).power


def _pump_lo(platform: CalibrationPlatform, pump: ChannelId) -> str:
    """LO of an IQ pump channel, holding its power."""
    if pump not in platform.channels:
        raise ValueError(f"TWPA pump channel {pump} not found in the platform.")
    lo = getattr(platform.channels[pump], "lo", None)
    if lo is None:
        raise ValueError(f"TWPA pump channel {pump} has no LO, holding its power.")
    return lo


def _edges(gain: npt.NDArray, i: int, j: int) -> list[str]:
    """Swept ranges whose edge contains the grid point `(i, j)`.

    Ranges with a single value are not swept, thus they have no edge.
    """
    edges = []
    for name, index, size in (
        ("amplitude", i, gain.shape[0]),
        ("frequency", j, gain.shape[1]),
    ):
        if size > 1 and index in (0, size - 1):
            edges.append(name)
    return edges


def _select(gain: npt.NDArray, neighbourhood: int) -> tuple[int, int, bool]:
    """Select the operating point on the gain grid.

    The point maximizes the minimum gain over its neighbourhood. If the maximum lies at
    the edge of the scanned ranges, it is returned as a candidate.

    Returns:
        The amplitude and frequency indices, and whether the point is a candidate.
    """
    smoothed = minimum_filter(gain, size=2 * neighbourhood + 1, mode="nearest")
    i, j = (int(index) for index in np.unravel_index(np.argmax(smoothed), gain.shape))
    return i, j, bool(_edges(gain, i, j))


def _fit(
    data: TwpaFrequencyOffsetData,
    fitpars: TwpaFrequencyOffsetFitParameters | None = None,
) -> TwpaFrequencyOffsetResults:
    """Select a robust operating point, within the gain plateau.

    After aggregating the gain across the evaluated probes, as in the `objective` fit
    parameter (by default, their minimum), select the TWPA frequency and offset as in
    :func:`_select`. The maximum gain may be just below the over-pumping region, where
    the gain drops sharply, and small drifts of the pump cause large gain losses:
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
        i, j, is_candidate = _select(gain_map, fitpars.neighbourhood)
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


def _probe_label(data: TwpaFrequencyOffsetData, index: int) -> str:
    """Label of the probe `index`, by its frequency if known."""
    if data.probes is None:
        return "Readout probe"
    return f"Probe {data.probes[index] * nano:.4f} GHz"


def _probe_labels(data: TwpaFrequencyOffsetData, target: QubitId) -> list[str]:
    """Labels of the probes amplified by the pump of `target`.

    With several targets on the pump, they are prefixed by the target.
    """
    probes = data.group_probes(target)
    if len(data.group(target)) == 1:
        return [_probe_label(data, p) for _, p in probes]
    if data.probes is None:
        return [f"Qubit {q} readout" for q, _ in probes]
    return [f"Qubit {q} {_probe_label(data, p)}" for q, p in probes]


def _gain_figure(
    data: TwpaFrequencyOffsetData,
    fit: TwpaFrequencyOffsetResults | None,
    target: QubitId,
    gain: npt.NDArray,
    title: str,
) -> go.Figure:
    """Heatmap of a gain map, with the points of maximum gain and of the fit.

    To relate the dimensionless amplitude offset to the effective pump power
    in dB, a secondary vertical axis is rendered on the right edge. Its tick positions
    and labels are generated dynamically from the platform pump power and the logarithmic
    offset scaling. Because Plotly requires an associated trace to render secondary layout
    axes, an invisible dummy scatter trace is attached to this secondary scale.
    """
    fig = go.Figure()
    base_power = data.power.get(target, 0.0)

    offsets = np.array(data.offset[target])
    frequencies = np.array(data.frequency[target]) * nano
    valid_mask = np.abs(offsets) > 1e-12
    tickvals = offsets[valid_mask]
    if len(tickvals) > 8:
        indices = np.linspace(0, len(tickvals) - 1, 8, dtype=int)
        tickvals = tickvals[indices]
    powers = base_power + 20 * np.log10(np.abs(tickvals))
    ticktext = [f"{np.round(power, 1)}" for power in powers]

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=data.offset[target],
            z=gain,
            colorscale="inferno",
            colorbar_x=1.15,
            colorbar_title_text=title,
        ),
    )
    # Invisible trace required for Plotly to render the secondary y-axis (yaxis2).
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 0, "opacity": 0},
            hoverinfo="skip",
            showlegend=False,
            yaxis="y2",
        )
    )
    i, j = np.unravel_index(np.argmax(gain), gain.shape)
    fig.add_trace(
        go.Scatter(
            x=[frequencies[j]],
            y=[offsets[i]],
            mode="markers",
            marker={
                "size": 12,
                "color": "deepskyblue",
                "symbol": "diamond-open",
                "line": {"width": 2},
            },
            name="Maximum Gain",
            showlegend=True,
        )
    )
    if fit is not None and target in fit:
        candidate = fit.is_candidate(target)
        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target] * nano],
                y=[fit.offset[target]],
                mode="markers",
                marker=(
                    {
                        "size": 16,
                        "color": "magenta",
                        "symbol": "circle-x-open",
                        "line": {"width": 2},
                    }
                    if candidate
                    else {"size": 10, "color": "black", "symbol": "cross"}
                ),
                name="Candidate Point (scan edge)" if candidate else "Optimal Point",
                showlegend=True,
            )
        )
    fig.update_xaxes(title_text="Pump Frequency [GHz]")
    fig.update_yaxes(title_text="Pump Amplitude")
    if len(frequencies) > 1:
        df = abs(frequencies[1] - frequencies[0]) / 2
        fig.update_xaxes(range=[np.min(frequencies) - df, np.max(frequencies) + df])
    if len(offsets) > 1:
        doff = abs(offsets[1] - offsets[0]) / 2
        fig.update_yaxes(range=[np.min(offsets) - doff, np.max(offsets) + doff])
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "y": -0.25},
        yaxis2={
            "title_text": "Pump power [dB]",
            "overlaying": "y",
            "side": "right",
            "matches": "y",
            "tickmode": "array",
            "tickvals": tickvals.tolist(),
            "ticktext": ticktext,
            "showgrid": False,
        },
    )
    return fig


PROBE_COLUMNS = 3
"""Maximum number of columns of the per-probe gain maps."""


def _probes_figure(
    data: TwpaFrequencyOffsetData,
    fit: TwpaFrequencyOffsetResults | None,
    target: QubitId,
) -> go.Figure:
    """Gain maps of each probe, on a common color scale, with the point of the fit."""
    gain = data.probe_gain(target)
    nprobes = gain.shape[2]
    cols = min(nprobes, PROBE_COLUMNS)
    rows = -(-nprobes // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=_probe_labels(data, target),
    )
    frequencies = np.array(data.frequency[target]) * nano
    zmin, zmax = float(np.min(gain)), float(np.max(gain))
    for p in range(nprobes):
        row, col = p // cols + 1, p % cols + 1
        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=data.offset[target],
                z=gain[:, :, p],
                zmin=zmin,
                zmax=zmax,
                coloraxis="coloraxis",
            ),
            row=row,
            col=col,
        )
        if fit is not None and target in fit:
            fig.add_trace(
                go.Scatter(
                    x=[fit.frequency[target] * nano],
                    y=[fit.offset[target]],
                    mode="markers",
                    marker={"size": 8, "color": "black", "symbol": "cross"},
                    name="Selected Point",
                    showlegend=p == 0,
                ),
                row=row,
                col=col,
            )
    fig.update_layout(
        coloraxis={"colorscale": "inferno", "colorbar_title_text": "TWPA Gain [dB]"},
        legend={"orientation": "h", "y": -0.15},
        height=300 * rows + 100,
    )
    fig.update_xaxes(title_text="Pump Frequency [GHz]", row=rows)
    fig.update_yaxes(title_text="Pump Amplitude", col=1)
    return fig


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
    shared = (
        f"<p>The TWPA pump <code>{data.pump[target]}</code> is shared by qubits "
        f"{', '.join(str(q) for q in group)}: the gain is aggregated over all their "
        f"probes, and a single operating point is selected for the pump.</p>"
        if len(group) > 1
        else ""
    )
    figures = []
    if target == group[0]:
        figures.append(
            _gain_figure(
                data,
                fit,
                target,
                data.gain(target, objective),
                OBJECTIVE_LABELS[objective],
            )
        )
    else:
        shared += f"<p>Gain maps are reported for qubit {group[0]}.</p>"
    nprobes = data.probe_gain(target).shape[2]
    if nprobes > 1 and target == group[0]:
        figures += [
            _gain_figure(
                data, fit, target, data.gain(target, other), OBJECTIVE_LABELS[other]
            )
            for other in others
        ]
        figures.append(_probes_figure(data, fit, target))

    if fit is not None and target in fit and fit.is_candidate(target):
        i = data.offset[target].index(fit.offset[target])
        j = data.frequency[target].index(fit.frequency[target])
        edges = " and ".join(
            f"pump {edge}" for edge in _edges(data.gain(target, objective), i, j)
        )
        fitting_report = shared + (
            f"<p><b>Fit failed</b>: the operating point lies at the edge of the scanned "
            f"{edges} range, thus the optimum may lie outside of it. Extend the scan "
            f"(e.g. changing the base pump power) and repeat it. The candidate point "
            f"is marked in the plot, and it is not applied.</p>"
        )
    elif fit is not None and target in fit:
        labels = [
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
            probe_labels = _probe_labels(data, target)
            if len(probe_labels) != len(probe_gain):
                # results not matching the probes grouping of the data
                probe_labels = [f"Probe {p}" for p in range(len(probe_gain))]
            labels += [f"{label} Gain [dB]" for label in probe_labels]
            values += [np.round(value, 4) for value in probe_gain]
        fitting_report = shared + table_html(
            table_dict([target] * len(labels), labels, values)
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

    A standalone oscillator pump holds both in its configuration. Instead, an IQ pump
    channel holds the frequency in its own configuration, and the power in its LO.

    The LO frequency is left unchanged, since the scan is performed at fixed LO: moving
    it would also move the LO leakage and the sideband image, which alter the pumping
    conditions, and thus the gain, of the selected point.

    Targets sharing the same pump are assigned the same point by the fit, thus their
    updates coincide.

    Candidate points are not results, thus there is nothing to apply.
    """
    if results.is_candidate(target):
        return
    pump = platform.channels[platform.qubits[target].acquisition].twpa_pump
    assert pump is not None
    frequency, power = results.frequency[target], results.power[target]
    power_config = _power_config(platform, pump)
    platform.update(
        {f"configs.{pump}.frequency": frequency, f"configs.{power_config}.power": power}
    )


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot, _update)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object."""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
