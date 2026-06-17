"""Spin-locking (T1rho) noise power spectral density spectroscopy.

The qubit is prepared on the equator with a Y90 pulse, locked along the
rotating-frame X axis by a continuous square drive of variable duration and
amplitude, and finally back-projected onto the Z axis with a second Y90
pulse. Sweeping the spin-lock duration at fixed amplitude yields the
longitudinal relaxation rate :math:`\\Gamma_{1\\rho}`, while sweeping the
amplitude maps :math:`\\Gamma_{1\\rho}` onto the effective Rabi frequency
:math:`\\nu_R`, providing a measurement of the environmental noise power
spectral density :math:`S(\\nu_R)`.

References:
    - F. Yan et al., `Rotating-frame relaxation as a noise spectrum analyser
      of a superconducting qubit undergoing driven evolution
      <https://www.nature.com/articles/ncomms3337>`_, Nat. Commun. 4, 2337
      (2013).
    - F. Yan et al., `The flux qubit revisited to enhance coherence and
      reproducibility <https://www.nature.com/articles/ncomms12964>`_, Nat.
      Commun. 7, 12964 (2016).
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    LongPulse,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import GHZ_TO_HZ, table_dict, table_html
from .utils import exp_decay, single_exponential_fit

__all__ = ["SpinLockParameters", "SpinLockResults", "spin_lock"]


@dataclass
class SpinLockParameters(Parameters):
    """SpinLock runcard inputs."""

    duration_min: int
    """Minimum spin-lock pulse duration [ns]."""
    duration_max: int
    """Maximum spin-lock pulse duration [ns]."""
    duration_step: int
    """Step spin-lock pulse duration [ns]."""
    amplitude_min: float
    """Minimum spin-lock pulse amplitude [a.u.]."""
    amplitude_max: float
    """Maximum spin-lock pulse amplitude [a.u.]."""
    n_amplitudes: int
    """Number of amplitude points, log-spaced between amplitude_min and amplitude_max."""
    sequence: Literal["SL3", "SL5a", "SL5b"] = "SL3"
    """Spin-lock sequence type."""

    @property
    def duration_range(self) -> tuple[int, int, int]:
        """Return a tuple with the spin-lock pulse duration range."""
        return self.duration_min, self.duration_max, self.duration_step


@dataclass
class SpinLockResults(Results):
    """SpinLock outputs."""

    t1rho: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """:math:`T_{1\\rho}` and its error for each spin-lock amplitude [ns]."""
    gamma: dict[QubitId, list[float]] = field(default_factory=dict)
    """Relaxation rate :math:`\\Gamma_{1\\rho} = S(\\nu_R)` for each amplitude [Hz]."""
    gamma_error: dict[QubitId, list[float]] = field(default_factory=dict)
    """Error on :math:`\\Gamma_{1\\rho}` [Hz]."""
    rabi_frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """Effective Rabi frequency :math:`\\nu_R` for each amplitude [Hz]."""
    fitted_parameters: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Raw exponential fit parameters for each amplitude."""


SpinLockType = np.dtype(
    [
        ("duration", np.float64),
        ("amp", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for spin-lock routine."""


@dataclass
class SpinLockData(Data):
    """SpinLock acquisition outputs."""

    pi_pulse_duration: dict[QubitId, float] = field(default_factory=dict)
    """RX pulse duration used to convert amplitudes into Rabi frequencies [ns]."""
    pi_pulse_amplitude: dict[QubitId, float] = field(default_factory=dict)
    """RX pulse amplitude used to convert amplitudes into Rabi frequencies."""
    data: dict[QubitId, npt.NDArray[SpinLockType]] = field(default_factory=dict)
    """Raw data acquired."""

    def durations(self, qubit: QubitId) -> npt.NDArray:
        """Unique spin-lock durations for a given qubit."""
        return np.unique(self[qubit].duration)

    def amplitudes(self, qubit: QubitId) -> npt.NDArray:
        """Unique spin-lock amplitudes for a given qubit."""
        return np.unique(self[qubit].amp)

    def rabi_frequency(self, qubit: QubitId) -> npt.NDArray:
        """Effective Rabi frequency [Hz] associated to each spin-lock amplitude.

        Obtained by rescaling the swept amplitudes with the Rabi rate
        extracted from the calibrated RX (pi) pulse.
        """
        return (
            self.amplitudes(qubit)
            / self.pi_pulse_amplitude[qubit]
            / (2 * self.pi_pulse_duration[qubit])
            * GHZ_TO_HZ
        )


def _acquisition(
    params: SpinLockParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SpinLockData:
    """Data acquisition for the spin-lock (T1rho) experiment."""

    duration_range = np.arange(*params.duration_range)
    amplitude_range = np.logspace(
        np.log10(params.amplitude_min),
        np.log10(params.amplitude_max),
        params.n_amplitudes,
    )

    # Build per-qubit fixed infrastructure once: Y90, echo, readout pulse, and
    # calibration metadata.  These objects are reused across every amplitude
    # iteration so their UUIDs stay constant and result lookup stays valid.
    ro_pulses: dict[QubitId, Pulse] = {}
    pi_pulse_duration: dict[QubitId, float] = {}
    pi_pulse_amplitude: dict[QubitId, float] = {}
    _infra: dict[QubitId, tuple] = {} # pulse infrastructure: (y90, echo, qd_channel, ro_channel, ro_pulse)

    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel = platform.qubits[q].drive
        ro_channel, ro_pulse = natives.MZ()[0]
        rx_pulse = natives.RX()[0][1]
        pi_pulse_duration[q] = rx_pulse.duration
        pi_pulse_amplitude[q] = rx_pulse.amplitude
        ro_pulses[q] = ro_pulse
        y90 = natives.R(theta=np.pi / 2, phi=np.pi / 2)
        echo = (
            natives.R(theta=np.pi, phi=0 if params.sequence == "SL5a" else np.pi)
            if params.sequence in ("SL5a", "SL5b")
            else PulseSequence()
        )
        _infra[q] = (y90, echo, qd_channel, ro_channel, ro_pulse)

    data = SpinLockData(
        pi_pulse_duration=pi_pulse_duration,
        pi_pulse_amplitude=pi_pulse_amplitude,
    )

    # Outer Python loop over log-spaced amplitudes.  Each amplitude maps to one
    # independent T1ρ decay curve, so we do one hardware execution per
    # amplitude and accumulate the duration sweep results row by row.
    for amp in amplitude_range:
        sequence = PulseSequence()
        pulse_to_sweep: dict[QubitId, LongPulse] = {}
        ro_delays: dict[QubitId, Delay] = {}

        for q in targets:
            y90, echo, qd_channel, ro_channel, ro_pulse = _infra[q]
            spin_lock_pulse = LongPulse(
                duration=duration_range[0],
                amplitude=amp,
                relative_phase=np.pi / 2, # Lock along the rotating-frame X axis
            )
            ro_delay = Delay(duration=duration_range[0])

            sequence += y90
            sequence += echo
            sequence.append((qd_channel, spin_lock_pulse))
            sequence += echo
            sequence += y90
            sequence.append(
                (ro_channel, Delay(duration=2 * y90.duration + 2 * echo.duration))
            )
            sequence.append((ro_channel, ro_delay))
            sequence.append((ro_channel, ro_pulse))

            pulse_to_sweep[q] = spin_lock_pulse
            ro_delays[q] = ro_delay

        # The LongPulse sweeper is kept separate from the ro_delay sweeper
        # because the Qblox backend applies a (duration - 4) offset to the
        # LongPulse wait register, which must not be applied to the Delay.
        duration_sweeper: ParallelSweepers = [
            Sweeper(
                parameter=Parameter.duration,
                values=duration_range,
                pulses=[pulse_to_sweep[q]],
            )
            for q in targets
        ] + [
            Sweeper(
                parameter=Parameter.duration,
                values=duration_range,
                pulses=[ro_delays[q]],
            )
            for q in targets
        ]

        results = platform.execute(
            [sequence],
            [duration_sweeper],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        )

        for q in targets:
            prob = probability(results[ro_pulses[q].id], state=1)
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                SpinLockType,
                q,
                dict(
                    duration=duration_range,
                    amp=np.full(len(duration_range), amp),
                    prob=prob,
                    error=error,
                ),
            )

    return data


def _fit(data: SpinLockData) -> SpinLockResults:
    """Extract :math:`\\Gamma_{1\\rho}(\\nu_R)` from per-amplitude exponential decays."""

    t1rho = {}
    gamma = {}
    gamma_error = {}
    rabi_frequency = {}
    fitted_parameters = {}

    for qubit in data.qubits:
        durations = data.durations(qubit)
        amplitudes = data.amplitudes(qubit)
        qubit_data = data[qubit]
        probs = qubit_data.prob.reshape(len(amplitudes), len(durations))
        errors = qubit_data.error.reshape(len(amplitudes), len(durations))

        t1rho[qubit] = []
        gamma[qubit] = []
        gamma_error[qubit] = []
        fitted_parameters[qubit] = []
        for amplitude, prob, error in zip(amplitudes, probs, errors):
            try:
                decay, popt, _, _ = single_exponential_fit(durations, prob, error)
                t1rho[qubit].append(decay)
                gamma[qubit].append(GHZ_TO_HZ / decay[0])
                gamma_error[qubit].append(GHZ_TO_HZ * decay[1] / decay[0] ** 2)
                fitted_parameters[qubit].append(popt)
            except Exception as e:
                log.warning(
                    f"Spin-lock T1rho fit failed for qubit {qubit} and "
                    f"amplitude {amplitude} due to {e}."
                )
                t1rho[qubit].append([np.nan, np.nan])
                gamma[qubit].append(np.nan)
                gamma_error[qubit].append(np.nan)
                fitted_parameters[qubit].append([np.nan, np.nan, np.nan])

        rabi_frequency[qubit] = data.rabi_frequency(qubit).tolist()

    return SpinLockResults(t1rho, gamma, gamma_error, rabi_frequency, fitted_parameters)


def _plot(data: SpinLockData, target: QubitId, fit: SpinLockResults = None):
    """Plotting for the spin-lock (T1rho / noise PSD) experiment."""

    figures = []
    fitting_report = ""

    durations = data.durations(target)
    amplitudes = data.amplitudes(target)
    qubit_data = data[target]
    probs = qubit_data.prob.reshape(len(amplitudes), len(durations))

    fig_map = go.Figure(
        go.Heatmap(
            x=durations,
            y=amplitudes,
            z=probs,
            colorbar=dict(title="P(1)"),
        )
    )
    fig_map.update_layout(
        xaxis_title="Spin-lock duration [ns]",
        yaxis_title="Spin-lock amplitude [a.u.]",
        title="T1ρ decay map",
    )
    figures.append(fig_map)

    if fit is not None:
        fig_decay = go.Figure()
        indices = np.linspace(0, len(amplitudes) - 1, min(5, len(amplitudes))).astype(
            int
        )
        duration_range = np.linspace(
            durations.min(), durations.max(), 2 * len(durations)
        )
        for i in indices:
            amplitude = amplitudes[i]
            color = f"hsl({int(360 * i / max(len(amplitudes), 1))}, 70%, 45%)"
            fig_decay.add_trace(
                go.Scatter(
                    x=durations,
                    y=probs[i],
                    mode="markers",
                    name=f"A={amplitude:.4f}",
                    legendgroup=f"A={amplitude:.4f}",
                    marker=dict(color=color),
                )
            )
            popt = fit.fitted_parameters[target][i]
            if not np.any(np.isnan(popt)):
                fig_decay.add_trace(
                    go.Scatter(
                        x=duration_range,
                        y=exp_decay(duration_range, *popt),
                        mode="lines",
                        line=dict(dash="dot", color=color),
                        name=f"Fit A={amplitude:.4f}",
                        legendgroup=f"A={amplitude:.4f}",
                        showlegend=False,
                    )
                )
        fig_decay.update_layout(
            xaxis_title="Spin-lock duration [ns]",
            yaxis_title="Probability of state 1",
            title="T1ρ decay curves",
        )
        figures.append(fig_decay)

        rabi_freq = np.array(fit.rabi_frequency[target])
        gamma = np.array(fit.gamma[target])
        gamma_error = np.array(fit.gamma_error[target])
        mask = np.isfinite(gamma) & (gamma > 0) & (rabi_freq > 0)

        fig_psd = go.Figure(
            go.Scatter(
                x=rabi_freq[mask],
                y=gamma[mask],
                error_y=dict(array=gamma_error[mask]),
                mode="markers+lines",
                name="S(ν_R)",
            )
        )
        fig_psd.update_layout(
            xaxis_title="Effective Rabi frequency ν_R [Hz]",
            yaxis_title="Γ₁ρ = S(ν_R) [Hz]",
            xaxis_type="log",
            yaxis_type="log",
            title="Noise power spectral density",
        )
        figures.append(fig_psd)

        if np.any(mask):
            fitting_report = table_html(
                table_dict(
                    target,
                    [
                        "Rabi frequency range [Hz]",
                        "Γ₁ρ range [Hz]",
                    ],
                    [
                        f"{rabi_freq[mask].min():.3e} - {rabi_freq[mask].max():.3e}",
                        f"{gamma[mask].min():.3e} - {gamma[mask].max():.3e}",
                    ],
                )
            )

    return figures, fitting_report


spin_lock = Routine(_acquisition, _fit, _plot)
"""SpinLock Routine object."""
