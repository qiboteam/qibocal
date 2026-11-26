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

The single shots are acquired, and reduced to their mean and spread, which tell the
fluctuations expected from the shot noise alone: those in excess are drifts. The phase
of the signals tells their kind, cf. :func:`_diagnosis`.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode

from ...auto.operation import Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import table_dict, table_html
from .utils import (
    TwpaData,
    TwpaProbeParameters,
    off_updates,
    probe_label,
    probe_updates,
    pump_point,
    pumps,
    readouts,
    reference,
    reference_shots,
    shots,
    statistics,
)

DB = 20 / np.log(10)
"""Relative change of an amplitude, corresponding to 1 dB, at first order."""

EXCESS = 2.0
"""Ratio of the observed to the shot-noise fluctuations, beyond which they are drifts."""

CORRELATED = 0.7
"""Correlation of the signals with the pump on and off, beyond which they drift together."""


@dataclass(kw_only=True)
class TwpaStabilityParameters(TwpaProbeParameters):
    """Inputs of the TWPA stability validation."""

    repeats: int = 10
    """Repetitions of the TWPA-off reference and of the TWPA-on signal."""
    frequency: float | None = None
    """Pump frequency [Hz] of the operating point.

    By default, the configured one.
    """
    offset: float = 1.0
    """Pump offset of the operating point, scaling the configured pump power.

    The default is the configured pump power, unchanged.
    """
    interval: float = 0.0
    """Wait [s] between the repetitions, to probe slower drifts."""

    def __post_init__(self):
        super().__post_init__()
        if self.repeats < 2:
            raise ValueError("At least two repetitions are needed to assess stability.")
        if self.interval < 0:
            raise ValueError("The interval between the repetitions must be >= 0.")


@dataclass
class TwpaStabilityData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Statistics of the signal with the TWPA on, ``(repeats, nprobes, 3)``.

    The last axis holds the mean I and Q, and the spread of the shots, cf.
    :func:`.utils.statistics`.
    """
    reference_value: dict[QubitId, list] = field(default_factory=dict)
    """Statistics of the signal with the TWPA off, with the shape of :attr:`data`."""
    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Pump frequency [Hz] of the operating point."""
    offset: float = 1.0
    """Pump offset of the operating point."""
    time: list[float] = field(default_factory=list)
    """Time [s] of each repetition, since the first one."""
    shots: int = 1
    """Shots of each acquisition with the TWPA on."""
    reference_shots: int = 1
    """Shots of each acquisition with the TWPA off."""

    def nprobes(self, qubit: QubitId) -> int:
        """Number of probes evaluated for `qubit`."""
        return self.data[qubit].shape[1]

    def reference_array(self, qubit: QubitId) -> npt.NDArray:
        """Statistics of the signal with the TWPA off, ``(repeats, nprobes, 3)``."""
        return np.array(self.reference_value[qubit]).reshape(self.data[qubit].shape)

    def _magnitudes(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Magnitudes of the signal with the TWPA on and off, ``(repeats, nprobes)``."""
        on = magnitude(self[qubit][..., :2])
        off = magnitude(self.reference_array(qubit)[..., :2])
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

    def phase_fluctuations(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Fluctuations [deg] of the phase of the signal with the TWPA on and off.

        They are taken around the circular mean of the phases of each probe, thus they
        are not affected by the wrapping of the phase. The shapes are
        ``(repeats, nprobes)``.
        """

        def around_mean(stats: npt.NDArray) -> npt.NDArray:
            iq = stats[..., 0] + 1j * stats[..., 1]
            unit = iq / np.abs(iq)
            return np.degrees(np.angle(unit / np.mean(unit, axis=0)))

        return around_mean(self[qubit]), around_mean(self.reference_array(qubit))

    def shot_noise(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Relative fluctuations of the mean signal expected from the shot noise alone.

        It is the spread of the shots over the magnitude of their mean, scaled by the
        square root of their number, and averaged over the repetitions: the relative
        standard deviation of both the magnitude and the phase [rad] of the mean, at
        first order. The shapes are ``(nprobes,)``, for the TWPA on and off.
        """

        def relative(stats: npt.NDArray, nshots: int) -> npt.NDArray:
            return np.mean(
                stats[..., 2] / magnitude(stats[..., :2]) / np.sqrt(nshots), axis=0
            )

        return (
            relative(self[qubit], self.shots),
            relative(self.reference_array(qubit), self.reference_shots),
        )

    def reference_snr(self, qubit: QubitId) -> npt.NDArray:
        """Single-shot SNR [dB] of the signal with the TWPA off, ``(nprobes,)``."""
        off = self.reference_array(qubit)
        return np.mean(20 * np.log10(magnitude(off[..., :2]) / off[..., 2]), axis=0)


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
    pumped_excess: dict[QubitId, list[float]]
    """Ratio of :attr:`pumped_std` to the one expected from the shot noise alone."""
    reference_excess: dict[QubitId, list[float]]
    """Ratio of :attr:`reference_std` to the one expected from the shot noise alone."""
    pumped_phase_std: dict[QubitId, list[float]]
    """Standard deviation [deg] of the phase of the signal with the TWPA on."""
    reference_phase_std: dict[QubitId, list[float]]
    """Standard deviation [deg] of the phase of the signal with the TWPA off."""
    reference_phase_excess: dict[QubitId, list[float]]
    """Ratio of :attr:`reference_phase_std` to the one expected from the shot noise."""
    correlation: dict[QubitId, list[float]]
    """Correlation of the fluctuations of the signal with the TWPA on and off."""
    reference_snr: dict[QubitId, list[float]]
    """Single-shot SNR [dB] of the signal with the TWPA off."""
    diagnosis: dict[QubitId, list[str]] = field(default_factory=dict)
    """What limits the stability of the gain of each probe, cf. :func:`_diagnosis`."""


def _acquisition(
    params: TwpaStabilityParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaStabilityData:
    """Repeat the TWPA-off reference and the TWPA-on signal at a single point.

    Each repetition acquires the reference first, as a scan does, and then the signal
    with the pump at the operating point, for each probe. The pump plays the point as a
    continuous wave, as during the readout, cf. :func:`.utils.pump_point`. The single
    shots are reduced to their statistics as soon as downloaded.
    """
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    updates = probe_updates(platform, targets, params.probes, params.detunings)
    off = off_updates(platform, pump.unique, params.off_power)
    frequency, point = pump_point(platform, pump, params.frequency, params.offset)
    nshots, nreference = shots(params, platform), reference_shots(params, platform)

    on: dict[QubitId, list[list[npt.NDArray]]] = {q: [] for q in targets}
    off_: dict[QubitId, list[list[npt.NDArray]]] = {q: [] for q in targets}
    times = []
    start = time.monotonic()
    for repetition in range(params.repeats):
        if repetition > 0 and params.interval > 0:
            time.sleep(params.interval)
        times.append(time.monotonic() - start)
        signal: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
        references: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
        for probe in updates:
            pumped_off = reference(
                platform,
                sequence,
                handles,
                [probe],
                off,
                nreference,
                params.relaxation_time,
                AveragingMode.SINGLESHOT,
            )[0]
            results = platform.execute(
                [sequence],
                nshots=nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.SINGLESHOT,
                updates=[probe, point],
            )
            for qubit in targets:
                references[qubit].append(statistics(pumped_off[qubit]))
                signal[qubit].append(statistics(results[handles[qubit]]))
        for qubit in targets:
            on[qubit].append(signal[qubit])
            off_[qubit].append(references[qubit])

    return TwpaStabilityData(
        data={q: np.array(on[q]) for q in targets},
        reference_value={q: np.array(off_[q]).tolist() for q in targets},
        frequency=frequency,
        offset=params.offset,
        time=times,
        shots=nshots,
        reference_shots=nreference,
        probes=params.probes,
        detunings=params.detunings,
        power=pump.power,
        pump=pump.channel,
    )


def _correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Correlation of two series, null if either does not fluctuate."""
    if np.isclose(np.ptp(x), 0) or np.isclose(np.ptp(y), 0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _diagnosis(
    pumped_excess: float,
    reference_excess: float,
    pumped_std: float,
    reference_std: float,
    correlation: float,
    reference_phase_excess: float,
) -> str:
    """What limits the stability of the gain, from the fluctuations of the signals.

    Fluctuations within :data:`EXCESS` times the shot noise are just noise. Beyond it,
    signals drifting together (:data:`CORRELATED`) point to a common cause, e.g. the
    readout chain, while the largest drift otherwise limits the gain. The phase of the
    reference drifting beyond the shot noise hints at a moving resonance: on the
    resonance, its frequency shifts the phase linearly, but the magnitude only at second
    order.
    """
    pumped = pumped_excess > EXCESS
    referenced = reference_excess > EXCESS
    if not pumped and not referenced:
        verdict = "shot noise"
    elif pumped and referenced and correlation > CORRELATED:
        verdict = "common drift (readout chain, or resonator)"
    elif referenced and (not pumped or reference_std > pumped_std):
        verdict = "reference drift"
    else:
        verdict = "pumped signal drift (pump, or TWPA)"
    if reference_phase_excess > EXCESS:
        verdict += "; reference phase drift (moving resonance?)"
    return verdict


def _fit(data: TwpaStabilityData) -> TwpaStabilityResults:
    """Statistics of the gain and of the signals with the TWPA on and off."""

    def std(values: npt.NDArray) -> npt.NDArray:
        return np.std(values, axis=0, ddof=1)

    results: dict[str, dict[QubitId, list]] = {
        f: {}
        for f in (
            "gain",
            "gain_std",
            "common_gain_std",
            "pumped_std",
            "reference_std",
            "pumped_excess",
            "reference_excess",
            "pumped_phase_std",
            "reference_phase_std",
            "reference_phase_excess",
            "correlation",
            "reference_snr",
            "diagnosis",
        )
    }
    for qubit in data.qubits:
        on, off = data.fluctuations(qubit)
        phase_on, phase_off = data.phase_fluctuations(qubit)
        noise_on, noise_off = data.shot_noise(qubit)
        stats = {
            "gain": np.mean(data.gain(qubit), axis=0),
            "gain_std": std(data.gain(qubit)),
            "common_gain_std": std(data.common_gain(qubit)),
            "pumped_std": std(on),
            "reference_std": std(off),
            "pumped_excess": std(on) / (DB * noise_on),
            "reference_excess": std(off) / (DB * noise_off),
            "pumped_phase_std": std(phase_on),
            "reference_phase_std": std(phase_off),
            "reference_phase_excess": std(phase_off) / np.degrees(noise_off),
            "correlation": np.array(
                [_correlation(on[:, p], off[:, p]) for p in range(on.shape[1])]
            ),
            "reference_snr": data.reference_snr(qubit),
        }
        for name, values in stats.items():
            results[name][qubit] = values.tolist()
        results["diagnosis"][qubit] = [
            _diagnosis(
                stats["pumped_excess"][p],
                stats["reference_excess"][p],
                stats["pumped_std"][p],
                stats["reference_std"][p],
                stats["correlation"][p],
                stats["reference_phase_excess"][p],
            )
            for p in range(on.shape[1])
        ]
    return TwpaStabilityResults(**results)


REPORT = (
    ("Mean gain [dB]", "gain"),
    ("Gain std [dB]", "gain_std"),
    ("Gain std, common reference [dB]", "common_gain_std"),
    ("TWPA-on signal std [dB]", "pumped_std"),
    ("TWPA-off signal std [dB]", "reference_std"),
    ("TWPA-on excess over shot noise", "pumped_excess"),
    ("TWPA-off excess over shot noise", "reference_excess"),
    ("TWPA-on phase std [deg]", "pumped_phase_std"),
    ("TWPA-off phase std [deg]", "reference_phase_std"),
    ("TWPA-off phase excess over shot noise", "reference_phase_excess"),
    ("On/off correlation", "correlation"),
    ("TWPA-off single-shot SNR [dB]", "reference_snr"),
)
"""Rows of the report, with the corresponding results fields."""


def _plot(data: TwpaStabilityData, fit: TwpaStabilityResults | None, target: QubitId):
    """Plot the gain and the signals with the TWPA on and off, over time.

    The first figure compares the gain of each repetition with respect to its own
    reference, as a scan measures it, and to the mean reference of all of them. The
    second one shows how the magnitude and the phase of the signals with the TWPA on
    and off fluctuate, each around its own mean, with the band expected from the shot
    noise alone. They tell which of the two limits the stability of the gain.
    """
    nprobes = data.nprobes(target)
    labels = [probe_label(data, p) for p in range(nprobes)]
    times = np.array(data.time)
    on, off = data.fluctuations(target)
    phase_on, phase_off = data.phase_fluctuations(target)
    noise_on, noise_off = data.shot_noise(target)

    gain = go.Figure()
    for p, label in enumerate(labels):
        for name, values, dash in (
            ("own reference", data.gain(target), None),
            ("common reference", data.common_gain(target), "dot"),
        ):
            gain.add_trace(
                go.Scatter(
                    x=times,
                    y=values[:, p],
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"{label}, {name}",
                )
            )
    gain.update_layout(
        showlegend=True,
        xaxis_title="Time [s]",
        yaxis_title="TWPA Gain [dB]",
        legend={"orientation": "h", "y": -0.25},
    )

    signals = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[
            "TWPA on, magnitude [dB]",
            "TWPA on, phase [deg]",
            "TWPA off (reference), magnitude [dB]",
            "TWPA off (reference), phase [deg]",
        ],
    )
    panels = (
        (1, 1, on, DB * noise_on),
        (1, 2, phase_on, np.degrees(noise_on)),
        (2, 1, off, DB * noise_off),
        (2, 2, phase_off, np.degrees(noise_off)),
    )
    for row, col, values, noise in panels:
        for p, label in enumerate(labels):
            signals.add_trace(
                go.Scatter(
                    x=times,
                    y=values[:, p],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=row == 1 and col == 1,
                ),
                row=row,
                col=col,
            )
            # the band expected from the shot noise alone, at one standard deviation
            signals.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate(
                        [np.full_like(times, noise[p]), np.full_like(times, -noise[p])]
                    ),
                    fill="toself",
                    opacity=0.2,
                    line={"width": 0},
                    hoverinfo="skip",
                    name=f"{label}, shot noise",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    signals.update_xaxes(title_text="Time [s]", row=2)
    signals.update_layout(legend={"orientation": "h", "y": -0.2})

    report = ""
    if fit is not None and target in fit.gain:
        columns, values = [], []
        for name, attribute in REPORT:
            columns += [f"{label} {name}" for label in labels]
            values += [np.round(v, 4) for v in getattr(fit, attribute)[target]]
        columns += [f"{label} Diagnosis" for label in labels]
        values += fit.diagnosis.get(target, [])
        report = table_html(table_dict([target] * len(columns), columns, values))
    return [gain, signals], report


twpa_stability = Protocol(_acquisition, _fit, _plot)
"""Stability of a TWPA operating point, over repeated references and signals."""
