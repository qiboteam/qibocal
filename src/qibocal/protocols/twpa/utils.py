"""Building blocks shared by the TWPA protocols.

The protocols themselves are independent: each one defines its own parameters, data,
results and acquisition, since they trade different quantities for acquisition time and
memory. Only the plumbing they have in common lives here:

- the discovery of the pump driving the readout of each target, and of its power;
- the readout sequence, and the probe and TWPA-off updates;
- the pump-off reference acquisition;
- the selection of a robust operating point on a 2D map;
- the update of the pump, and the figures displaying maps over its frequency and offset.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

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
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
)
from scipy.constants import nano
from scipy.ndimage import minimum_filter

from ...auto.operation import Data, QubitId
from ...calibration.platform import CalibrationPlatform
from ..utils import readout_frequency

PUMP_OFF_POWER = -60.0
"""Pump power [dB] of the TWPA-off reference.

Low enough to leave the TWPA unpumped, while within the range of both standalone
oscillators and controllers' ports, in the instruments it has been tuned on. It can
be overridden through the ``off_power`` parameter of the protocols.
"""


def power_config(platform: CalibrationPlatform, pump: ChannelId) -> str:
    """Configuration holding the pump power.

    The pump is either a standalone oscillator, whose configuration holds the power,
    or an IQ channel of a controller, whose power is set by its LO.
    """
    if isinstance(platform.config(pump), OscillatorConfig):
        return pump
    return pump_lo(platform, pump)


def pump_power(platform: CalibrationPlatform, pump: ChannelId) -> float:
    """Configured pump power."""
    return cast(OscillatorConfig, platform.config(power_config(platform, pump))).power


def pump_lo(platform: CalibrationPlatform, pump: ChannelId) -> str:
    """LO of an IQ pump channel, holding its power."""
    if pump not in platform.channels:
        raise ValueError(f"TWPA pump channel {pump} not found in the platform.")
    lo = getattr(platform.channels[pump], "lo", None)
    if lo is None:
        raise ValueError(f"TWPA pump channel {pump} has no LO, holding its power.")
    return lo


@dataclass
class Pumps:
    """TWPA pumps amplifying the readout of the targets."""

    channel: dict[QubitId, ChannelId]
    """Pump channel of each target."""
    config: dict[QubitId, OscillatorConfig | IqConfig]
    """Pump configuration of each target."""
    power: dict[QubitId, float]
    """Configured base pump power [dB] of each target."""

    @property
    def unique(self) -> dict[ChannelId, QubitId]:
        """Distinct pump channels, each associated to the first target it amplifies.

        Targets may share a pump: it has to be swept, or turned off, only once.
        """
        unique: dict[ChannelId, QubitId] = {}
        for qubit, channel in self.channel.items():
            unique.setdefault(channel, qubit)
        return unique


def pumps(platform: CalibrationPlatform, targets: Sequence[QubitId]) -> Pumps:
    """Pumps amplifying the readout of `targets`."""
    channel = {}
    config = {}
    power = {}
    for qubit in targets:
        acquisition = platform.qubits[qubit].acquisition
        if acquisition is None:
            raise ValueError(f"Acquisition channel for qubit {qubit} not defined.")
        pump = platform.channels[acquisition].twpa_pump
        if pump is None:
            raise ValueError(
                f"Qubit {qubit} does not have a TWPA pump channel configured."
            )
        channel[qubit] = pump
        config[qubit] = cast(OscillatorConfig | IqConfig, platform.config(pump))
        power[qubit] = pump_power(platform, pump)
    return Pumps(channel=channel, config=config, power=power)


def readouts(
    platform: CalibrationPlatform,
    targets: Sequence[QubitId],
    amplitude: float,
    duration: float,
) -> tuple[PulseSequence, dict[QubitId, str]]:
    """Simultaneous rectangular readouts of the targets.

    Returns:
        The sequence, and the identifier of the acquisition of each target in the
        results of its execution.
    """
    acquisition = {}
    for qubit in targets:
        acq = platform.qubits[qubit].acquisition
        if acq is None:
            raise ValueError(f"Acquisition channel for qubit {qubit} not defined.")
        acquisition[qubit] = acq
    readout = {
        qubit: Readout(
            probe=Pulse(amplitude=amplitude, duration=duration, envelope=Rectangular()),
            acquisition=Acquisition(duration=duration),
        )
        for qubit in targets
    }
    sequence = PulseSequence([(acquisition[q], readout[q]) for q in targets])
    return sequence, {q: readout[q].acquisition.id for q in targets}


def probe_updates(
    platform: CalibrationPlatform,
    targets: Sequence[QubitId],
    probes: Sequence[float] | None,
) -> list[dict[ChannelId, dict[str, float]]]:
    """Platform updates setting the probes frequencies, one per probe evaluated.

    When `probes` is omitted, there is a single probe, where each target is probed at
    its own calibrated readout frequency.
    """
    if probes is not None:
        return [
            {platform.qubits[q].probe: {"frequency": probe} for q in targets}
            for probe in probes
        ]
    return [
        {
            platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
            for q in targets
        }
    ]


def off_updates(
    platform: CalibrationPlatform,
    channels: Iterable[ChannelId],
    off_power: float | None = None,
) -> dict[str, dict[str, float]]:
    """Platform update turning the pumps off, by lowering their power.

    It also suppresses the LO leakage, unlike a null offset.
    """
    power = PUMP_OFF_POWER if off_power is None else off_power
    return {power_config(platform, ch): {"power": power} for ch in channels}


def reference(
    platform: CalibrationPlatform,
    sequence: PulseSequence,
    handles: Mapping[QubitId, str],
    updates: Sequence[dict[ChannelId, dict[str, float]]],
    off: dict[str, dict[str, float]],
    nshots: int,
    relaxation_time: float,
    averaging_mode: AveragingMode,
) -> list[dict[QubitId, npt.NDArray]]:
    """Pump-off reference, for each of the probes updates.

    Returns:
        For each probe, the acquisition of each target, as returned by the platform.
    """
    results = []
    for probe in updates:
        execution = platform.execute(
            [sequence],
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=averaging_mode,
            updates=[probe, off],
        )
        results.append({q: execution[handle] for q, handle in handles.items()})
    return results


@dataclass
class TwpaData(Data):
    """Data of the protocols amplifying many targets by shared pumps.

    The targets of a pump are amplified by the same working point, thus their probes
    are all aggregated, and a single point is selected for the pump.
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

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        raise NotImplementedError

    def group(self, qubit: QubitId) -> list[QubitId]:
        """Targets amplified by the same pump as `qubit`, in acquisition order."""
        pump = self.pump.get(qubit)
        if pump is None:
            return [qubit]
        return [q for q in self.qubits if self.pump.get(q) == pump]

    def group_probes(self, qubit: QubitId) -> list[tuple[QubitId, int]]:
        """Probes amplified by the pump of `qubit`, as ``(target, probe index)``."""
        return [(q, p) for q in self.group(qubit) for p in range(self.nprobes(q))]


def probe_label(probes: Sequence[float] | None, index: int) -> str:
    """Label of the probe `index`, by its frequency if known."""
    if probes is None:
        return "Readout probe"
    return f"Probe {probes[index] * nano:.4f} GHz"


def probe_labels(
    probes: Sequence[float] | None,
    group: Sequence[QubitId],
    group_probes: Sequence[tuple[QubitId, int]],
) -> list[str]:
    """Labels of the probes amplified by a pump.

    With several targets on the pump, they are prefixed by the target.
    """
    if len(group) == 1:
        return [probe_label(probes, p) for _, p in group_probes]
    if probes is None:
        return [f"Qubit {q} readout" for q, _ in group_probes]
    return [f"Qubit {q} {probe_label(probes, p)}" for q, p in group_probes]


def edges(shape: tuple[int, int], i: int, j: int) -> list[str]:
    """Swept ranges whose edge contains the grid point `(i, j)`.

    Ranges with a single value are not swept, thus they have no edge.
    """
    return [
        name
        for name, index, size in (
            ("amplitude", i, shape[0]),
            ("frequency", j, shape[1]),
        )
        if size > 1 and index in (0, size - 1)
    ]


def select(grid: npt.NDArray, neighbourhood: int) -> tuple[int, int, bool]:
    """Select the operating point on a grid to be maximized.

    The point maximizes the minimum of the grid over its neighbourhood. If the maximum
    lies at the edge of the scanned ranges, it is returned as a candidate.

    Returns:
        The amplitude and frequency indices, and whether the point is a candidate.
    """
    smoothed = minimum_filter(grid, size=2 * neighbourhood + 1, mode="nearest")
    i, j = (int(index) for index in np.unravel_index(np.argmax(smoothed), grid.shape))
    return i, j, bool(edges(grid.shape, i, j))


def update_pump(
    platform: CalibrationPlatform,
    target: QubitId,
    frequency: float,
    power: float,
):
    """Set the pump amplifying `target` to a frequency and power.

    A standalone oscillator pump holds both in its configuration. Instead, an IQ pump
    channel holds the frequency in its own configuration, and the power in its LO.

    The LO frequency is left unchanged, since the scans are performed at fixed LO:
    moving it would also move the LO leakage and the sideband image, which alter the
    pumping conditions, and thus the gain, of the selected point.
    """
    pump = platform.channels[platform.qubits[target].acquisition].twpa_pump
    assert pump is not None
    platform.update(
        {
            f"configs.{pump}.frequency": frequency,
            f"configs.{power_config(platform, pump)}.power": power,
        }
    )


POWER_AXIS_WIDTH = 70
"""Width [px] of the pump power axis, ticks and title, on the right of the heatmap."""

MARGIN = {"t": 30}
"""Figure margins [px], with no title on top."""

PROBE_COLUMNS = 3
"""Maximum number of columns of the per-probe maps."""


def colorbar(title: str, xpad: int = 10) -> dict:
    """Colorbar of a map, as narrow as possible.

    The title runs vertically along the bar, instead of on top of it, where its length
    would reserve a wide margin, shrinking the heatmap. The bar is placed `xpad` pixels
    past the plot area, independently of the figure width.
    """
    return {
        "title": {"text": title, "side": "right"},
        "x": 1,
        "xref": "paper",
        "xanchor": "left",
        "xpad": xpad,
        "thickness": 20,
    }


@dataclass
class Selected:
    """Operating point marked on the maps."""

    frequency: float
    """Pump frequency [Hz]."""
    offset: float
    """Pump offset."""
    candidate: bool = False
    """Whether it is only a candidate, lying at the edge of the scanned ranges."""


def map_figure(
    frequency: Sequence[float],
    offset: Sequence[float],
    base_power: float,
    z: npt.NDArray,
    title: str,
    selected: Selected | None = None,
    maximum: str = "Maximum Gain",
) -> go.Figure:
    """Heatmap of a map over the pump frequency and offset.

    The point of maximum and the selected one are marked. To relate the dimensionless
    amplitude offset to the effective pump power in dB, a secondary vertical axis is
    rendered on the right edge. Its tick positions and labels are generated dynamically
    from the platform pump power and the logarithmic offset scaling. Because Plotly
    requires an associated trace to render secondary layout axes, an invisible dummy
    scatter trace is attached to this secondary scale.

    Args:
        frequency: pump frequencies [Hz] of the columns of `z`.
        offset: pump offsets of the rows of `z`.
        base_power: configured base pump power [dB].
        z: the map, with shape ``(len(offset), len(frequency))``.
        title: the title of the colorbar.
        selected: the point selected by the fit, if any.
        maximum: the name of the point of maximum.
    """
    fig = go.Figure()
    offsets = np.array(offset)
    frequencies = np.array(frequency) * nano
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
            y=offsets,
            z=z,
            colorscale="inferno",
            # past the pump power axis, on its right
            colorbar=colorbar(title, xpad=POWER_AXIS_WIDTH),
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
    i, j = np.unravel_index(np.argmax(z), z.shape)
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
            name=maximum,
            showlegend=True,
        )
    )
    if selected is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected.frequency * nano],
                y=[selected.offset],
                mode="markers",
                marker=(
                    {
                        "size": 16,
                        "color": "magenta",
                        "symbol": "circle-x-open",
                        "line": {"width": 2},
                    }
                    if selected.candidate
                    else {"size": 10, "color": "black", "symbol": "cross"}
                ),
                name=(
                    "Candidate Point (scan edge)"
                    if selected.candidate
                    else "Optimal Point"
                ),
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
        margin=MARGIN,
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


def probes_figure(
    frequency: Sequence[float],
    offset: Sequence[float],
    maps: npt.NDArray,
    labels: Sequence[str],
    title: str,
    selected: Selected | None = None,
) -> go.Figure:
    """Maps of each probe, on a common color scale, with the selected point.

    Args:
        frequency: pump frequencies [Hz] of the columns of the maps.
        offset: pump offsets of the rows of the maps.
        maps: the maps, with shape ``(len(offset), len(frequency), nprobes)``.
        labels: the titles of the maps of each probe.
        title: the title of the colorbar.
        selected: the point selected by the fit, if any.
    """
    nprobes = maps.shape[2]
    cols = min(nprobes, PROBE_COLUMNS)
    rows = -(-nprobes // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=list(labels),
    )
    frequencies = np.array(frequency) * nano
    zmin, zmax = float(np.min(maps)), float(np.max(maps))
    for p in range(nprobes):
        row, col = p // cols + 1, p % cols + 1
        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=offset,
                z=maps[:, :, p],
                zmin=zmin,
                zmax=zmax,
                coloraxis="coloraxis",
            ),
            row=row,
            col=col,
        )
        if selected is not None:
            fig.add_trace(
                go.Scatter(
                    x=[selected.frequency * nano],
                    y=[selected.offset],
                    mode="markers",
                    marker={"size": 8, "color": "black", "symbol": "cross"},
                    name="Selected Point",
                    showlegend=p == 0,
                ),
                row=row,
                col=col,
            )
    fig.update_layout(
        coloraxis={"colorscale": "inferno", "colorbar": colorbar(title)},
        legend={"orientation": "h", "y": -0.15},
        margin=MARGIN,
        height=300 * rows + 100,
    )
    fig.update_xaxes(title_text="Pump Frequency [GHz]", row=rows)
    fig.update_yaxes(title_text="Pump Amplitude", col=1)
    return fig


def shared_note(data: TwpaData, target: QubitId, quantity: str = "gain") -> str:
    """Note on the aggregation of the probes of the targets sharing a pump.

    `quantity` is the one aggregated across the probes, e.g. the gain.
    """
    group = data.group(target)
    if len(group) == 1:
        return ""
    return (
        f"<p>The TWPA pump <code>{data.pump[target]}</code> is shared by qubits "
        f"{', '.join(str(q) for q in group)}: the {quantity} is aggregated over all "
        f"their probes, and a single operating point is selected for the pump.</p>"
    )
