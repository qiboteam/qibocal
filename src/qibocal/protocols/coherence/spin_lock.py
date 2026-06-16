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
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import GHZ_TO_HZ, table_dict, table_html
from .utils import exp_decay, single_exponential_fit
MAX_WIDTH = 2**13


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
    amplitude_step: float
    """Step spin-lock pulse amplitude [a.u.]."""
    sequence: Literal["SL2", "SL5a", "SL5b"] = "SL2"
    """Spin-lock sequence type."""

    @property
    def duration_range(self) -> tuple[int, int, int]:
        """Return a tuple with the spin-lock pulse duration range."""
        return self.duration_min, self.duration_max, self.duration_step

    @property
    def amplitude_range(self) -> tuple[float, float, float]:
        """Return a tuple with the spin-lock pulse amplitude range."""
        return self.amplitude_min, self.amplitude_max, self.amplitude_step
    
    @property
    def n_pulses(self) -> int:
        """Minimum N such that each sub-pulse duration fits within MAX_WIDTH samples."""
        return max(1, -(-self.duration_max // MAX_WIDTH))

    @property
    def coerced_duration_step(self) -> int:
        """``duration_step`` rounded to the nearest multiple of ``n_pulses``.

        Guarantees that sub-pulse step ``coerced_duration_step // n_pulses`` is
        a positive integer, removing any constraint on the user-supplied step.
        """
        n = self.n_pulses
        if n <= 1:
            return self.duration_step
        return max(n, round(self.duration_step / n) * n)

    @property
    def extra_duration(self) -> int:
        """Fixed extra duration [ns] appended after the N sub-pulses.

        Equal to ``duration_min % n_pulses``.  Because ``coerced_duration_step``
        is a multiple of ``n_pulses``, this remainder is constant across all
        sweep steps, so a single fixed ``Delay`` is sufficient.
        """
        n = self.n_pulses
        return self.duration_min % n if n > 1 else 0

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

    amplitude_range = np.arange(*params.amplitude_range)
    n = params.n_pulses
    extra_duration = params.extra_duration
    coerced_step = params.coerced_duration_step

    # Effective total durations using the coerced step (uniform multiple of n).
    actual_duration_values = np.arange(
        params.duration_min, params.duration_max, coerced_step
    )
    short_duration_values = ((actual_duration_values - extra_duration) // n).astype(int)

    sequence = PulseSequence()
    pulse_to_sweep: dict[QubitId, Pulse] = {}
    ro_delays: dict[QubitId, Delay] = {}
    ro_pulses: dict[QubitId, Pulse] = {}
    pi_pulse_duration: dict[QubitId, float] = {}
    pi_pulse_amplitude: dict[QubitId, float] = {}

    # Build the reusable native pulses/sequences once per qubit: the Y90
    # prepare/back-projection rotation, the spin-lock square pulse, and the
    # optional refocusing pi pulse for the SL5a/SL5b echo variants.
    pulses_store: dict[QubitId, tuple[PulseSequence, Pulse, PulseSequence]] = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        # One pulse object shared across all n repetitions. 
        spin_lock_pulse = Pulse(
            duration=int(short_duration_values[0]),
            amplitude=amplitude_range[0],
            relative_phase=0,
            envelope=Rectangular(),
        )
        echo = (
            natives.R(theta=np.pi, phi=0 if params.sequence == "SL5a" else np.pi)
            if params.sequence in ("SL5a", "SL5b")
            else PulseSequence()
        )
        pulses_store[qubit] = (
            natives.R(theta=np.pi / 2, phi=np.pi / 2),
            spin_lock_pulse,
            echo,
        )
        pulse_to_sweep[qubit] = spin_lock_pulse

    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel = platform.qubits[q].drive
        ro_channel, ro_pulse = natives.MZ()[0]

        rx_pulse = natives.RX()[0][1]
        pi_pulse_duration[q] = rx_pulse.duration
        pi_pulse_amplitude[q] = rx_pulse.amplitude

        y90, spin_lock_pulse, echo = pulses_store[q]
        ro_delay = Delay(duration=int(actual_duration_values[0]))

        # Y90: bring |0> to the equator, aligned with the X axis
        sequence += y90
        # Optional refocusing pi pulse before the spin-lock drive (SL5a/SL5b)
        sequence += echo
        # Play the same pulse object n times (to share UUID)
        for _ in range(n):
            sequence.append((qd_channel, spin_lock_pulse))
        # Fixed extra delay so total drive time == duration_min for the first
        # step (and all steps when duration_step % n == 0).
        if extra_duration > 0:
            sequence.append((qd_channel, Delay(duration=extra_duration)))
        # Optional refocusing pi pulse after the spin-lock drive (SL5a/SL5b)
        sequence += echo
        # Y90: back-project the locked state onto the Z axis for readout
        sequence += y90
        sequence.append(
            (ro_channel, Delay(duration=2 * y90.duration + 2 * echo.duration))
        )
        sequence.append((ro_channel, ro_delay))
        sequence.append((ro_channel, ro_pulse))

        ro_delays[q] = ro_delay
        ro_pulses[q] = ro_pulse

    # The sub-pulse sweeper and the ro_delay sweeper are kept separate.
    duration_sweeper: ParallelSweepers = [
        Sweeper(
            parameter=Parameter.duration,
            values=short_duration_values,
            pulses=[pulse_to_sweep[q]],
        )
        for q in targets
    ] + [
        Sweeper(
            parameter=Parameter.duration,
            values=actual_duration_values,
            pulses=[ro_delays[q]],
        )
        for q in targets
    ]

    amplitude_sweeper: ParallelSweepers = [
        Sweeper(
            parameter=Parameter.amplitude,
            values=amplitude_range,
            pulses=[pulse_to_sweep[q]],
        )
        for q in targets
    ]

    results = platform.execute(
        [sequence],
        [amplitude_sweeper, duration_sweeper],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    data = SpinLockData(
        pi_pulse_duration=pi_pulse_duration,
        pi_pulse_amplitude=pi_pulse_amplitude,
    )

    duration_mesh, amplitude_mesh = np.meshgrid(actual_duration_values, amplitude_range)
    for q in targets:
        prob = probability(results[ro_pulses[q].id], state=1)
        error = np.sqrt(prob * (1 - prob) / params.nshots)
        data.register_qubit(
            SpinLockType,
            (q),
            dict(
                duration=duration_mesh.ravel(),
                amp=amplitude_mesh.ravel(),
                prob=prob.ravel(),
                error=error.ravel(),
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
