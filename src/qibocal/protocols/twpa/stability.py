"""Protocol to validate the stability of a TWPA operating point.

The gain of a TWPA is the ratio of the readout signal with the pump on and off. It is
measured against a TWPA-off reference, acquired along with each scan. If that reference,
or the pump, drifts between acquisitions, the gain of the very same working point
changes from scan to scan, as large as tens of dB, even though the point is reproducible
and its gain map is not. Scanning the pump parameters cannot tell: the map has the
structure of the amplifier, and its level the one of the drift.

This protocol repeats, at a single operating point, the pair of acquisitions of a scan:
the TWPA-off reference, and the TWPA-on signal. It reports the fluctuations of the
two of them separately, and the gain with respect to the reference of each repetition,
and to a single common one. A gain whose fluctuations are mostly removed by the common
reference is limited by the reference, rather than by the pump.

Only the averages are needed, thus it is as cheap as :mod:`.frequency_offset`.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode

from ...auto.operation import Parameters, Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import table_dict, table_html
from .utils import (
    TwpaData,
    off_updates,
    offset_sweepers,
    probe_label,
    probe_updates,
    pumps,
    readouts,
    reference,
)


@dataclass
class TwpaStabilityParameters(Parameters):
    """Inputs of the TWPA stability validation."""

    repeats: int = 10
    """Repetitions of the TWPA-off reference and of the TWPA-on signal."""
    frequency: float | None = None
    """Pump frequency [Hz] of the operating point.

    By default, the configured one.
    """
    offset: float = 1.0
    """Pump offset of the operating point.

    The default is the configured pump power, unchanged.
    """
    probes: list[float] | None = None
    """List of probe frequencies to evaluate (Hz).

    If omitted, each target is probed exclusively at its own calibrated readout
    frequency.
    """
    probe_duration: float = 4e3
    """Probe wave duration."""
    probe_amplitude: float = 1.0
    """Probe wave amplitude."""
    off_power: float | None = None
    """Pump power [dB] of the TWPA-off reference, cf. :data:`.utils.PUMP_OFF_POWER`."""

    def __post_init__(self):
        if self.repeats < 2:
            raise ValueError("At least two repetitions are needed to assess stability.")


@dataclass
class TwpaStabilityData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Signal with the TWPA on, with shape ``(repeats, nprobes, 2)``."""
    reference_value: dict[QubitId, list[list[list[float]]]] = field(
        default_factory=dict
    )
    """Signal with the TWPA off, with the shape of :attr:`data`."""
    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Pump frequency [Hz] of the operating point."""
    offset: float = 1.0
    """Pump offset of the operating point."""
    time: list[float] = field(default_factory=list)
    """Time [s] of each repetition, since the first one."""

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.data[qubit].shape[1]

    def reference_array(self, qubit: QubitId) -> npt.NDArray:
        """Signal with the TWPA off, with shape ``(repeats, nprobes, 2)``."""
        return np.array(self.reference_value[qubit]).reshape(self.data[qubit].shape)

    def _magnitudes(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Magnitudes of the signal with the TWPA on and off, ``(repeats, nprobes)``."""
        on, off = magnitude(self[qubit]), magnitude(self.reference_array(qubit))
        if not np.all(np.isfinite(off)) or np.any(np.isclose(off, 0)):
            raise ValueError(
                f"TWPA-off reference magnitude for qubit {qubit} must be finite "
                "and nonzero, for every probe and repetition."
            )
        return on, off

    def gain(self, qubit: QubitId) -> npt.NDArray:
        """Gain [dB] of each repetition, with respect to its own reference."""
        on, off = self._magnitudes(qubit)
        return 20 * np.log10(on / off)

    def common_gain(self, qubit: QubitId) -> npt.NDArray:
        """Gain [dB] of each repetition, with respect to the mean reference."""
        on, off = self._magnitudes(qubit)
        return 20 * np.log10(on / np.mean(off, axis=0))

    def fluctuations(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Fluctuations [dB] of the signal with the TWPA on and off, around the mean.

        The shapes are ``(repeats, nprobes)``.
        """
        on, off = self._magnitudes(qubit)
        return (
            20 * np.log10(on / np.mean(on, axis=0)),
            20 * np.log10(off / np.mean(off, axis=0)),
        )


@dataclass
class TwpaStabilityResults(Results):
    gain: dict[QubitId, list[float]]
    """Mean gain [dB] of each probe, over the repetitions."""
    gain_std: dict[QubitId, list[float]]
    """Standard deviation [dB] of the gain of each probe, as the scan measures it."""
    common_gain_std: dict[QubitId, list[float]]
    """Standard deviation [dB] of the gain, if the reference were a single one."""
    pumped_std: dict[QubitId, list[float]]
    """Standard deviation [dB] of the signal with the TWPA on."""
    reference_std: dict[QubitId, list[float]]
    """Standard deviation [dB] of the signal with the TWPA off, the reference."""
    correlation: dict[QubitId, list[float]]
    """Correlation of the fluctuations of the signal with the TWPA on and off."""


def _acquisition(
    params: TwpaStabilityParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaStabilityData:
    """Repeat the TWPA-off reference and the TWPA-on signal at a single point.

    Each repetition acquires the reference first, as a scan does, and then the signal
    with the pump at the operating point, for each probe.
    """
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes)
    off = off_updates(platform, pump.unique, params.off_power)
    frequency = {
        q: pump.config[q].frequency if params.frequency is None else params.frequency
        for q in targets
    }
    # the operating point, applied as a scan does: the offset as a (single-valued) sweep
    point = {ch: {"frequency": frequency[q]} for ch, q in pump.unique.items()}
    sweepers = offset_sweepers(pump, np.array([params.offset]))

    on: dict[QubitId, list[list[npt.NDArray]]] = {q: [] for q in targets}
    off_: dict[QubitId, list[list[npt.NDArray]]] = {q: [] for q in targets}
    times = []
    start = time.monotonic()
    for _ in range(params.repeats):
        references = reference(
            platform,
            sequence,
            handles,
            updates,
            off,
            params.nshots,
            params.relaxation_time,
            AveragingMode.CYCLIC,
        )
        for qubit in targets:
            off_[qubit].append([probe[qubit] for probe in references])
        signal: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
        for probe in updates:
            results = platform.execute(
                [sequence],
                [sweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=[probe, point],
            )
            for qubit in targets:
                signal[qubit].append(np.reshape(results[handles[qubit]], 2))
        for qubit in targets:
            on[qubit].append(signal[qubit])
        times.append(time.monotonic() - start)

    return TwpaStabilityData(
        data={q: np.array(on[q]) for q in targets},
        reference_value={q: np.array(off_[q]).tolist() for q in targets},
        frequency=frequency,
        offset=params.offset,
        time=times,
        probes=params.probes,
        power=pump.power,
        pump=pump.channel,
    )


def _correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Correlation of two series, null if either does not fluctuate."""
    if np.isclose(np.ptp(x), 0) or np.isclose(np.ptp(y), 0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fit(data: TwpaStabilityData) -> TwpaStabilityResults:
    """Statistics of the gain and of the signals with the TWPA on and off."""

    def std(values: npt.NDArray) -> list[float]:
        return np.std(values, axis=0, ddof=1).tolist()

    gain, gain_std, common_std = {}, {}, {}
    pumped_std, reference_std, correlation = {}, {}, {}
    for qubit in data.qubits:
        gains = data.gain(qubit)
        on, off = data.fluctuations(qubit)
        gain[qubit] = np.mean(gains, axis=0).tolist()
        gain_std[qubit] = std(gains)
        common_std[qubit] = std(data.common_gain(qubit))
        pumped_std[qubit] = std(on)
        reference_std[qubit] = std(off)
        correlation[qubit] = [
            _correlation(on[:, p], off[:, p]) for p in range(on.shape[1])
        ]
    return TwpaStabilityResults(
        gain=gain,
        gain_std=gain_std,
        common_gain_std=common_std,
        pumped_std=pumped_std,
        reference_std=reference_std,
        correlation=correlation,
    )


def _plot(data: TwpaStabilityData, fit: TwpaStabilityResults | None, target: QubitId):
    """Plot the gain and the signals with the TWPA on and off, over the repetitions.

    The first figure compares the gain of each repetition with respect to its own
    reference, as a scan measures it, and to the mean reference of all of them. The
    second one shows how the signals with the TWPA on and off fluctuate, each around
    its own mean. They tell which of the two limits the stability of the gain.
    """
    nprobes = data.nprobes(target)
    labels = [probe_label(data.probes, p) for p in range(nprobes)]
    repeats = np.arange(1, len(data[target]) + 1)
    on, off = data.fluctuations(target)

    gain = go.Figure()
    for p, label in enumerate(labels):
        gain.add_trace(
            go.Scatter(
                x=repeats,
                y=data.gain(target)[:, p],
                mode="lines+markers",
                name=f"{label}, own reference",
            )
        )
        gain.add_trace(
            go.Scatter(
                x=repeats,
                y=data.common_gain(target)[:, p],
                mode="lines+markers",
                line={"dash": "dot"},
                name=f"{label}, common reference",
            )
        )
    gain.update_layout(
        showlegend=True,
        xaxis_title="Repetition",
        yaxis_title="TWPA Gain [dB]",
        legend={"orientation": "h", "y": -0.25},
    )

    signals = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["TWPA on", "TWPA off (reference)"],
    )
    for row, values in enumerate((on, off), start=1):
        for p, label in enumerate(labels):
            signals.add_trace(
                go.Scatter(
                    x=repeats,
                    y=values[:, p],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
    signals.update_xaxes(title_text="Repetition", row=2)
    signals.update_yaxes(title_text="Signal fluctuation [dB]")
    signals.update_layout(legend={"orientation": "h", "y": -0.2})

    report = ""
    if fit is not None and target in fit.gain:
        columns = []
        values = []
        for name, stat in (
            ("Mean gain [dB]", fit.gain),
            ("Gain std [dB]", fit.gain_std),
            ("Gain std, common reference [dB]", fit.common_gain_std),
            ("TWPA-on signal std [dB]", fit.pumped_std),
            ("TWPA-off signal std [dB]", fit.reference_std),
            ("On/off correlation", fit.correlation),
        ):
            columns += [f"{label} {name}" for label in labels]
            values += [np.round(v, 4) for v in stat[target]]
        report = table_html(table_dict([target] * len(columns), columns, values))
    return [gain, signals], report


twpa_stability = Protocol(_acquisition, _fit, _plot)
"""Stability of a TWPA operating point, over repeated references and signals."""
