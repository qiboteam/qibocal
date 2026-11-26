"""Protocol to measure the TWPA gain spectrum around the readout resonators.

The scans measure the gain at the readout frequency, which usually lies on the
resonance dip of the readout resonator: there the TWPA-off signal is the smallest, and
the most sensitive to any change of the resonance. If the dip moves between the
reference and the signal acquisitions (e.g. a fluctuating resonator), or if the pump
itself moves it (e.g. a pump leaking towards the resonator, or exciting the qubit), the
ratio of the two is not the gain of the amplifier, and it may be much larger than any
TWPA could give.

This protocol sweeps the probe frequency around the readout frequency of each target,
with the pump off and at an operating point, repeating the pair of spectra. It reports:

- the gain at the readout frequency, as the scans measure it;
- the baseline gain, away from the resonance, which is the one of the amplifier;
- their difference, the distortion of the gain by the resonance;
- the shift of the resonance caused by the pump, and its jitter across the repetitions;
- the depth of the readout frequency into the TWPA-off resonance.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from scipy.constants import mega, nano

from ...auto.operation import Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import RangeLike, readout_frequency, table_dict, table_html, to_range
from .utils import (
    TwpaData,
    TwpaReadoutParameters,
    off_updates,
    pump_point,
    pumps,
    range_values,
    readouts,
    reference_shots,
    shots,
)


@dataclass(kw_only=True)
class TwpaSpectrumParameters(TwpaReadoutParameters):
    """Inputs of the TWPA gain spectrum."""

    probe_frequency: RangeLike
    """Probe frequency [Hz] range, centered on the readout frequency of each target.

    It should cover the resonance, and enough of the transmission line around it, to
    estimate the baseline gain, cf. :attr:`TwpaSpectrumFitParameters.exclusion`.
    """
    repeats: int = 3
    """Repetitions of the pair of spectra, with the pump off and on."""
    frequency: float | None = None
    """Pump frequency [Hz] of the operating point. By default, the configured one."""
    offset: float = 1.0
    """Pump offset of the operating point, scaling the configured pump power."""
    interval: float = 0.0
    """Wait [s] between the repetitions, to probe slower drifts."""

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError("At least one repetition is needed.")
        if self.interval < 0:
            raise ValueError("The interval between the repetitions must be >= 0.")


@dataclass
class TwpaSpectrumFitParameters:
    """Analysis parameters of the TWPA gain spectrum."""

    exclusion: float = 2e6
    """Half-width [Hz] of the window around the TWPA-off resonance excluded from the
    baseline gain.

    It should be a few linewidths of the resonator, while leaving enough points in the
    scanned range.
    """

    def __post_init__(self):
        if self.exclusion < 0:
            raise ValueError("The exclusion half-width must be >= 0.")


@dataclass
class TwpaSpectrumData(TwpaData):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Signal with the TWPA on, with shape ``(repeats, nfrequencies, 2)``."""
    reference_value: dict[QubitId, list] = field(default_factory=dict)
    """Signal with the TWPA off, with the shape of :attr:`data`."""
    probe_frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """Probe frequencies [Hz] swept for each target."""
    readout: dict[QubitId, float] = field(default_factory=dict)
    """Readout frequency [Hz] of each target."""
    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Pump frequency [Hz] of the operating point."""
    offset: float = 1.0
    """Pump offset of the operating point."""
    time: list[float] = field(default_factory=list)
    """Time [s] of each repetition, since the first one."""

    def magnitudes(self, qubit: QubitId) -> tuple[npt.NDArray, npt.NDArray]:
        """Magnitudes with the TWPA on and off, ``(repeats, nfrequencies)``."""
        off = np.array(self.reference_value[qubit]).reshape(self.data[qubit].shape)
        return magnitude(self[qubit]), magnitude(off)

    def gain(self, qubit: QubitId) -> npt.NDArray:
        """Gain [dB] spectrum of each repetition, ``(repeats, nfrequencies)``."""
        on, off = self.magnitudes(qubit)
        return 20 * np.log10(on / off)


@dataclass
class TwpaSpectrumResults(Results):
    readout_gain: dict[QubitId, float]
    """Gain [dB] at the readout frequency, averaged over the repetitions."""
    readout_gain_std: dict[QubitId, float]
    """Standard deviation [dB] of the gain at the readout frequency."""
    baseline_gain: dict[QubitId, float]
    """Gain [dB] away from the resonance, averaged over the repetitions.

    It is the median over the frequencies outside the exclusion window, which is robust
    to residual features of the spectrum.
    """
    baseline_gain_std: dict[QubitId, float]
    """Standard deviation [dB] of the baseline gain."""
    distortion: dict[QubitId, float]
    """Gain [dB] at the readout frequency in excess of the baseline gain.

    It is caused by the resonance, and it should vanish: the TWPA gain varies slowly
    in the probe frequency, on the scale of the resonance.
    """
    dip: dict[QubitId, float]
    """Frequency [Hz] of the TWPA-off resonance, averaged over the repetitions."""
    dip_shift: dict[QubitId, float]
    """Shift [Hz] of the resonance caused by the pump, averaged over the repetitions."""
    dip_jitter: dict[QubitId, float]
    """Standard deviation [Hz] of the TWPA-off resonance, over the repetitions."""
    readout_depth: dict[QubitId, float]
    """Depth [dB] of the TWPA-off signal at the readout frequency, below the baseline.

    The deeper, the more sensitive the reference is to the resonance.
    """


def _acquisition(
    params: TwpaSpectrumParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaSpectrumData:
    """Repeat the spectra with the TWPA off and on, around each readout frequency.

    The probes of all the targets are swept in parallel, on board. In each repetition,
    the spectrum with the TWPA off is acquired first, as the reference of a scan, and
    then the one with the pump at the operating point, played as a continuous wave as
    during the readout, cf. :func:`.utils.pump_point`.
    """
    sequence, handles = readouts(
        platform, targets, params.probe_amplitude, params.probe_duration
    )
    pump = pumps(platform, targets)
    off = off_updates(platform, pump.unique, params.off_power)
    frequency, point = pump_point(platform, pump, params.frequency, params.offset)
    readout = {q: readout_frequency(q, platform) for q in targets}
    probe_frequency = {
        q: range_values(to_range(params.probe_frequency, center=readout[q]))
        for q in targets
    }
    # probes are swept in parallel: their sweeps have the same size, since the ranges
    # only differ by the center
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=np.array(probe_frequency[q]),
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    def spectrum(updates, nshots) -> dict[QubitId, npt.NDArray]:
        results = platform.execute(
            [sequence],
            [sweepers],
            nshots=nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )
        return {q: results[handles[q]] for q in targets}

    on: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    off_: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}
    times = []
    start = time.monotonic()
    for repetition in range(params.repeats):
        if repetition > 0 and params.interval > 0:
            time.sleep(params.interval)
        times.append(time.monotonic() - start)
        pumped_off = spectrum([off], reference_shots(params, platform))
        pumped = spectrum([point], shots(params, platform))
        for q in targets:
            off_[q].append(pumped_off[q])
            on[q].append(pumped[q])

    return TwpaSpectrumData(
        data={q: np.array(on[q]) for q in targets},
        reference_value={q: np.array(off_[q]).tolist() for q in targets},
        probe_frequency=probe_frequency,
        readout=readout,
        frequency=frequency,
        offset=params.offset,
        time=times,
        power=pump.power,
        pump=pump.channel,
    )


def dip(frequencies: npt.NDArray, magnitudes: npt.NDArray) -> npt.NDArray:
    """Frequency of the minimum of each spectrum, along the last axis.

    It is refined by the vertex of the parabola through the minimum and its neighbors,
    if it is not at the edge of the range.
    """
    index = np.argmin(magnitudes, axis=-1)
    inner = np.clip(index, 1, len(frequencies) - 2)
    rows = np.arange(magnitudes.shape[0])
    left, center, right = (magnitudes[rows, inner + k] for k in (-1, 0, 1))
    curvature = left - 2 * center + right
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(curvature > 0, 0.5 * (left - right) / curvature, 0.0)
    step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.0
    refined = frequencies[inner] + np.clip(vertex, -1, 1) * step
    edge = (index == 0) | (index == len(frequencies) - 1)
    return np.where(edge, frequencies[index], refined)


def _std(values: npt.NDArray) -> float:
    """Standard deviation over the repetitions, null for a single one."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _fit(
    data: TwpaSpectrumData, fitpars: TwpaSpectrumFitParameters | None = None
) -> TwpaSpectrumResults:
    """Gain at the readout frequency and away from the resonance, and its shift.

    The baseline gain is the median gain over the frequencies farther than
    `exclusion` from the TWPA-off resonance. If none is, it is not defined, and the
    target is skipped.
    """
    fitpars = TwpaSpectrumFitParameters() if fitpars is None else fitpars
    results: dict[str, dict[QubitId, float]] = {
        f: {}
        for f in (
            "readout_gain",
            "readout_gain_std",
            "baseline_gain",
            "baseline_gain_std",
            "distortion",
            "dip",
            "dip_shift",
            "dip_jitter",
            "readout_depth",
        )
    }
    for qubit in data.qubits:
        frequencies = np.array(data.probe_frequency[qubit])
        on, off = data.magnitudes(qubit)
        gain = data.gain(qubit)
        dips_off, dips_on = dip(frequencies, off), dip(frequencies, on)
        baseline = np.abs(frequencies - np.mean(dips_off)) > fitpars.exclusion
        if not np.any(baseline):
            continue
        readout = data.readout[qubit]
        readout_gain = np.array([np.interp(readout, frequencies, g) for g in gain])
        baseline_gain = np.median(gain[:, baseline], axis=1)
        readout_off = np.array([np.interp(readout, frequencies, o) for o in off])
        depth = 20 * np.log10(np.median(off[:, baseline], axis=1) / readout_off)
        stats = {
            "readout_gain": np.mean(readout_gain),
            "readout_gain_std": _std(readout_gain),
            "baseline_gain": np.mean(baseline_gain),
            "baseline_gain_std": _std(baseline_gain),
            "distortion": np.mean(readout_gain - baseline_gain),
            "dip": np.mean(dips_off),
            "dip_shift": np.mean(dips_on - dips_off),
            "dip_jitter": _std(dips_off),
            "readout_depth": np.mean(depth),
        }
        for name, value in stats.items():
            results[name][qubit] = float(value)
    return TwpaSpectrumResults(**results)


def _plot(data: TwpaSpectrumData, fit: TwpaSpectrumResults | None, target: QubitId):
    """Plot the spectra with the TWPA on and off, and the gain spectrum.

    The first figure shows the magnitude of the signal of each repetition, with the
    TWPA on and off, and the second one the gain spectrum, averaged over the
    repetitions, with its spread. The readout frequency, the TWPA-off resonance and the
    baseline gain are marked, if fitted.
    """
    frequencies = np.array(data.probe_frequency[target]) * nano
    on, off = data.magnitudes(target)
    gain = data.gain(target)
    readout = data.readout[target] * nano

    spectra = go.Figure()
    for r in range(on.shape[0]):
        for name, values, dash in (("TWPA off", off, "dot"), ("TWPA on", on, None)):
            spectra.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=20 * np.log10(values[r]),
                    mode="lines",
                    line={"dash": dash},
                    name=f"{name}, repetition {r + 1}",
                )
            )
    spectra.update_layout(
        xaxis_title="Probe frequency [GHz]",
        yaxis_title="Signal [dB]",
        legend={"orientation": "h", "y": -0.25},
    )

    mean, spread = (
        np.mean(gain, axis=0),
        (
            np.std(gain, axis=0, ddof=1)
            if gain.shape[0] > 1
            else np.zeros(gain.shape[1])
        ),
    )
    spectrum = go.Figure(
        [
            go.Scatter(
                x=np.concatenate([frequencies, frequencies[::-1]]),
                y=np.concatenate([mean + spread, (mean - spread)[::-1]]),
                fill="toself",
                opacity=0.3,
                line={"width": 0},
                hoverinfo="skip",
                name="Spread over repetitions",
            ),
            go.Scatter(x=frequencies, y=mean, mode="lines", name="Gain"),
        ]
    )
    for figure in (spectra, spectrum):
        figure.add_vline(
            x=readout, line_dash="dash", line_color="gray", annotation_text="readout"
        )

    report = ""
    if fit is not None and target in fit.baseline_gain:
        spectrum.add_hline(
            y=fit.baseline_gain[target],
            line_dash="dot",
            line_color="green",
            annotation_text="baseline",
        )
        spectrum.add_vline(
            x=fit.dip[target] * nano, line_dash="dot", line_color="black"
        )
        rows = (
            ("Gain at readout [dB]", fit.readout_gain[target]),
            ("Gain at readout std [dB]", fit.readout_gain_std[target]),
            ("Baseline gain [dB]", fit.baseline_gain[target]),
            ("Baseline gain std [dB]", fit.baseline_gain_std[target]),
            ("Resonance distortion of the gain [dB]", fit.distortion[target]),
            ("Resonance shift by the pump [MHz]", fit.dip_shift[target] / mega),
            ("Resonance jitter [MHz]", fit.dip_jitter[target] / mega),
            ("Readout depth into the resonance [dB]", fit.readout_depth[target]),
        )
        report = table_html(
            table_dict(
                [target] * len(rows),
                [name for name, _ in rows],
                [np.round(value, 4) for _, value in rows],
            )
        )
    spectrum.update_layout(
        xaxis_title="Probe frequency [GHz]",
        yaxis_title="TWPA Gain [dB]",
        legend={"orientation": "h", "y": -0.25},
    )
    return [spectra, spectrum], report


twpa_spectrum = Protocol(_acquisition, _fit, _plot)
"""TWPA gain spectrum around the readout resonators, with the pump off and on."""
