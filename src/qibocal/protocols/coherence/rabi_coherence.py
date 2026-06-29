"""Driven-evolution coherence spectroscopy (Rabi-coherence).

The qubit is driven on (or near) resonance at a fixed target Rabi frequency
:math:`\\Omega_R` for a variable interaction time :math:`\\tau`, and the
decay of the resulting Rabi oscillations is analysed in the rotating frame.
This protocol abstracts both flux-tunable transmons and persistent-current
(flux) qubits under a single generalised model, in which the qubit's lab
frame Hamiltonian is parametrised by a (possibly zero) longitudinal bias
term :math:`\\varepsilon` and a fixed transverse term :math:`\\Delta`, with
transition frequency :math:`\\omega_{01} = \\sqrt{\\varepsilon^2 + \\Delta^2}`.

The Rabi-decay envelope follows the generalised Bylander/Ithier expression

.. math::

    y(\\tau) = y_0 + A \\, \\zeta(\\tau) \\, e^{-\\Gamma_R \\tau}

with the quasi-static defocusing envelope

.. math::

    \\zeta(\\tau) = \\left[1 + (u\\tau)^2\\right]^{-1/4}
        \\cos\\left(2\\pi \\Omega_R \\tau + \\tfrac{1}{2}\\arctan(u\\tau)\\right),
    \\qquad u = \\left(\\frac{\\varepsilon}{\\omega_{01}}\\right)^2
        \\frac{\\sigma_\\varepsilon^2}{\\Omega_R},

and the exponential decay rate

.. math::

    \\Gamma_R = \\frac{3}{4}\\Gamma_1 \\cos^2\\theta
        + \\left(\\frac{\\varepsilon}{\\omega_{01}}\\right)^2
          \\frac{1}{2}\\Gamma_\\Omega,

where :math:`\\cos\\theta = \\Delta/\\omega_{01}` and :math:`\\Gamma_\\Omega`
is the noise power spectral density sampled at the Rabi frequency,
:math:`\\Gamma_\\Omega = \\pi S(\\Omega_R)`.

Throughout this module every frequency and rate is expressed in linear Hz,
matching the convention used by the rest of the software stack.  Conversion
to/from qibolab's native nanosecond pulse-duration convention is performed
explicitly wherever needed.

References:
    - G. Ithier et al., `Decoherence in a superconducting quantum bit
      circuit <https://arxiv.org/abs/cond-mat/0508588>`_, Phys. Rev. B 72,
      134519 (2005).
    - J. Bylander et al., `Dynamical decoupling and noise spectroscopy with
      a superconducting flux qubit <https://arxiv.org/abs/1101.4707>`_,
      Nat. Phys. 7, 565 (2011).
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Literal, Optional

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
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import chi2_reduced, table_dict, table_html

__all__ = ["RabiCoherenceParameters", "RabiCoherenceResults", "rabi_coherence"]

NS_TO_S = 1e-9
"""Conversion factor from nanoseconds (qibolab pulse convention) to seconds."""

QubitType = Literal["TRANSMON", "FLUX"]


@dataclass
class RabiCoherenceParameters(Parameters):
    """RabiCoherence runcard inputs."""

    rabi_frequency: float
    """Target Rabi frequency :math:`\\Omega_R` [Hz]."""
    delay_start: float
    """Initial drive/interaction time :math:`\\tau` [ns]."""
    delay_stop: float
    """Maximum drive/interaction time :math:`\\tau` [ns]."""
    delay_step: float
    """Step of the interaction time array [ns]."""
    flux_external: float = 0.0
    """External flux applied to the loop.

    For a ``TRANSMON`` this is interpreted as a flux-pulse amplitude [a.u.]
    fed into the qubit's calibrated ``detuning`` polynomial.  For a
    ``FLUX`` qubit this is the reduced external flux
    :math:`\\Phi_{\\text{ext}}/\\Phi_0` (with the symmetry point at 0.5).
    """
    artificial_detuning: float = 0.0
    """Intentional drive detuning :math:`\\delta'` [Hz]."""
    qubit_type: QubitType = "TRANSMON"
    """Qubit flavour: ``"TRANSMON"`` or ``"FLUX"``."""
    persistent_current: Optional[float] = None
    """Persistent current :math:`I_p` [Hz], required if ``qubit_type == "FLUX"``.

    Expressed directly in frequency units, i.e. such that
    :math:`\\varepsilon = 2 I_p (\\Phi_{\\text{ext}}/\\Phi_0 - 1/2)` is a
    frequency [Hz].
    """
    tunnel_splitting: Optional[float] = None
    """Tunnel splitting :math:`\\Delta` [Hz], required if ``qubit_type == "FLUX"``."""

    @property
    def delay_range(self) -> tuple[float, float, float]:
        """Return a tuple with the interaction-time sweep range."""
        return self.delay_start, self.delay_stop, self.delay_step

    def __post_init__(self):
        if self.qubit_type == "FLUX" and (
            self.persistent_current is None or self.tunnel_splitting is None
        ):
            raise ValueError(
                "`persistent_current` and `tunnel_splitting` must be provided "
                "when `qubit_type == 'FLUX'`."
            )


@dataclass
class RabiCoherenceResults(Results):
    """RabiCoherence outputs."""

    gamma_omega: dict[QubitId, float] = field(default_factory=dict)
    """Driven dephasing rate :math:`\\Gamma_\\Omega = \\pi S(\\Omega_R)` [Hz]."""
    gamma_omega_error: dict[QubitId, float] = field(default_factory=dict)
    """Error on :math:`\\Gamma_\\Omega` [Hz]."""
    sigma_epsilon: dict[QubitId, float] = field(default_factory=dict)
    """Quasi-static longitudinal noise standard deviation :math:`\\sigma_\\varepsilon` [Hz]."""
    sigma_epsilon_error: dict[QubitId, float] = field(default_factory=dict)
    """Error on :math:`\\sigma_\\varepsilon` [Hz]."""
    fitted_rabi_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Rabi frequency extracted from the fit [Hz]."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Raw fit parameters ``[y0, amplitude, rabi_frequency, gamma_omega, sigma_epsilon]``."""
    chi2: dict[QubitId, list[float]] = field(default_factory=dict)
    """Chi-squared reduced value and its expected error."""


RabiCoherenceType = np.dtype(
    [
        ("delay", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for the rabi-coherence routine."""


@dataclass
class RabiCoherenceData(Data):
    """RabiCoherence acquisition outputs."""

    qubit_type: QubitType = "TRANSMON"
    """Qubit flavour used for this acquisition."""
    rabi_frequency: float = 0.0
    """Target Rabi frequency :math:`\\Omega_R` [Hz]."""
    artificial_detuning: float = 0.0
    """Intentional drive detuning :math:`\\delta'` [Hz]."""
    pi_pulse_duration: dict[QubitId, float] = field(default_factory=dict)
    """RX pulse duration used to rescale the drive amplitude [ns]."""
    pi_pulse_amplitude: dict[QubitId, float] = field(default_factory=dict)
    """RX pulse amplitude used to rescale the drive amplitude."""
    drive_amplitude: dict[QubitId, float] = field(default_factory=dict)
    """Drive amplitude used to reach the target Rabi frequency."""
    omega01: dict[QubitId, float] = field(default_factory=dict)
    """Lab-frame transition frequency :math:`\\omega_{01}` [Hz]."""
    epsilon: dict[QubitId, float] = field(default_factory=dict)
    """Lab-frame longitudinal bias :math:`\\varepsilon` [Hz]."""
    cos_theta: dict[QubitId, float] = field(default_factory=dict)
    """Quantization-axis tilt factor :math:`\\cos\\theta = \\Delta/\\omega_{01}`."""
    gamma1: dict[QubitId, float] = field(default_factory=dict)
    """Relaxation rate :math:`\\Gamma_1 = 1/T_1` [Hz] from calibration."""
    data: dict[QubitId, npt.NDArray[RabiCoherenceType]] = field(default_factory=dict)
    """Raw data acquired."""

    def delays(self, qubit: QubitId) -> npt.NDArray:
        """Unique interaction-time values for a given qubit [ns]."""
        return np.unique(self[qubit].delay)

    def sensitivity(self, qubit: QubitId) -> float:
        """Noise-sensitivity factor :math:`(\\varepsilon/\\omega_{01})^2`."""
        omega01 = self.omega01[qubit]
        if omega01 == 0:
            return 0.0
        return (self.epsilon[qubit] / omega01) ** 2


def _qubit_frame(
    params: RabiCoherenceParameters,
    platform: CalibrationPlatform,
    qubit: QubitId,
) -> tuple[float, float, float]:
    """Compute ``(epsilon, omega01, cos_theta)`` in Hz for a given qubit.

    Strictly separates the lab-frame transition frequency ``omega01`` from
    the rotating-frame detuning, which is only computed downstream when
    choosing the drive frequency.
    """
    calibration = platform.calibration.single_qubits[qubit].qubit

    if params.qubit_type == "TRANSMON":
        omega01 = calibration.frequency_01
        if params.flux_external != 0:
            omega01 = omega01 + calibration.detuning(params.flux_external)
        epsilon = 0.0
        cos_theta = 1.0
        return epsilon, omega01, cos_theta

    # FLUX qubit
    ip = params.persistent_current
    delta = params.tunnel_splitting
    epsilon = 2 * ip * (params.flux_external - 0.5)
    omega01 = float(np.sqrt(epsilon**2 + delta**2))
    cos_theta = delta / omega01
    return epsilon, omega01, cos_theta


def _drive_amplitude(rx_amplitude: float, rx_duration: float, rabi_frequency: float) -> float:
    """Rescale the calibrated pi-pulse amplitude to reach ``rabi_frequency`` [Hz].

    ``rx_duration`` is expressed in ns (qibolab convention) and internally
    converted to seconds so that the returned amplitude matches a target
    Rabi frequency given in Hz.
    """
    t_rx = rx_duration * NS_TO_S
    return rx_amplitude * 2 * rabi_frequency * t_rx


def _acquisition(
    params: RabiCoherenceParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiCoherenceData:
    """Data acquisition for the rabi-coherence experiment."""

    delay_range = np.arange(*params.delay_range)

    data = RabiCoherenceData(
        qubit_type=params.qubit_type,
        rabi_frequency=params.rabi_frequency,
        artificial_detuning=params.artificial_detuning,
    )

    sequence = PulseSequence()
    pulse_to_sweep: dict[QubitId, LongPulse] = {}
    ro_delays: dict[QubitId, Delay] = {}
    ro_pulses = {}
    updates = []

    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel = platform.qubits[q].drive
        ro_channel, ro_pulse = natives.MZ()[0]
        rx_pulse = natives.RX()[0][1]

        epsilon, omega01, cos_theta = _qubit_frame(params, platform, q)
        drive_amplitude = _drive_amplitude(
            rx_pulse.amplitude, rx_pulse.duration, params.rabi_frequency
        )

        t1 = platform.calibration.single_qubits[q].t1
        if t1 is None or t1[0] is None:
            log.warning(
                f"T1 not available in calibration for qubit {q}: Gamma_1 set to 0."
            )
            gamma1 = 0.0
        else:
            gamma1 = 1 / (t1[0] * NS_TO_S)

        # On-resonance drive (rotating-frame detuning set to zero) plus the
        # intentionally introduced artificial detuning delta'.
        drive_frequency = omega01 + params.artificial_detuning
        updates.append({qd_channel: {"frequency": drive_frequency}})

        drive_pulse = LongPulse(
            duration=delay_range[0],
            amplitude=drive_amplitude,
            relative_phase=0,
        )
        ro_delay = Delay(duration=delay_range[0])

        sequence.append((qd_channel, drive_pulse))
        sequence.append((ro_channel, Delay(duration=drive_pulse.duration)))
        sequence.append((ro_channel, ro_delay))
        sequence.append((ro_channel, ro_pulse))

        pulse_to_sweep[q] = drive_pulse
        ro_delays[q] = ro_delay
        ro_pulses[q] = ro_pulse

        data.pi_pulse_duration[q] = rx_pulse.duration
        data.pi_pulse_amplitude[q] = rx_pulse.amplitude
        data.drive_amplitude[q] = drive_amplitude
        data.omega01[q] = omega01
        data.epsilon[q] = epsilon
        data.cos_theta[q] = cos_theta
        data.gamma1[q] = gamma1

    # As in the spin-lock routine, the LongPulse sweeper is kept separate
    # from the ro_delay sweeper because some backends apply a non trivial
    # offset to the LongPulse wait register that must not affect Delay.
    duration_sweeper: ParallelSweepers = [
        Sweeper(
            parameter=Parameter.duration,
            values=delay_range,
            pulses=[pulse_to_sweep[q]],
        )
        for q in targets
    ] + [
        Sweeper(
            parameter=Parameter.duration,
            values=delay_range,
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
        updates=updates,
    )

    for q in targets:
        prob = probability(results[ro_pulses[q].id], state=1)
        error = np.sqrt(prob * (1 - prob) / params.nshots)
        data.register_qubit(
            RabiCoherenceType,
            q,
            dict(
                delay=delay_range,
                prob=prob,
                error=error,
            ),
        )

    return data


def _zeta(tau: npt.NDArray, rabi_frequency: float, u: float) -> npt.NDArray:
    """Quasi-static defocusing envelope :math:`\\zeta(\\tau)`."""
    return (1 + (u * tau) ** 2) ** (-1 / 4) * np.cos(
        2 * np.pi * rabi_frequency * tau + 0.5 * np.arctan(u * tau)
    )


def _model(
    tau: npt.NDArray,
    y0: float,
    amplitude: float,
    rabi_frequency: float,
    gamma_omega: float,
    sigma_epsilon: float,
    gamma1: float,
    eta: float,
    cos_theta: float,
) -> npt.NDArray:
    """Driven-coherence decay model, all rates/frequencies in Hz, tau in seconds.

    ``eta = (epsilon / omega01)**2`` is fixed (not fitted) since it is fully
    determined by the qubit frame.  When ``eta == 0`` (transmon limit) the
    defocusing parameter ``u`` is forced to zero, recovering a pure
    exponentially damped cosine, avoiding any division by zero or
    ``arctan`` evaluated on an undefined quantity.
    """
    gamma_r = 0.75 * gamma1 * cos_theta**2 + eta * 0.5 * gamma_omega
    u = eta * sigma_epsilon**2 / rabi_frequency if eta > 0 and rabi_frequency > 0 else 0.0
    return y0 + amplitude * _zeta(tau, rabi_frequency, u) * np.exp(-gamma_r * tau)


def _fit(data: RabiCoherenceData) -> RabiCoherenceResults:
    """Extract :math:`\\Gamma_\\Omega(\\Omega_R)` from the driven-evolution decay."""

    gamma_omega = {}
    gamma_omega_error = {}
    sigma_epsilon = {}
    sigma_epsilon_error = {}
    fitted_rabi_frequency = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in data.qubits:
        delays_ns = data.delays(qubit)
        tau = delays_ns * NS_TO_S
        qubit_data = data[qubit]
        prob = qubit_data.prob
        error = qubit_data.error

        gamma1 = data.gamma1[qubit]
        cos_theta = data.cos_theta[qubit]
        eta = data.sensitivity(qubit)
        rabi_target = data.rabi_frequency

        y0_guess = float(np.mean(prob))
        amplitude_guess = float((np.max(prob) - np.min(prob)) / 2) or 0.1

        model = partial(_model, gamma1=gamma1, eta=eta, cos_theta=cos_theta)

        if eta > 0:
            sigma_guess = float(np.sqrt(rabi_target / eta)) if rabi_target > 0 else 1e6
            p0 = [y0_guess, amplitude_guess, rabi_target, gamma1, sigma_guess]
            bounds = (
                [-2, -2, 0, 0, 0],
                [2, 2, 5 * max(rabi_target, 1), np.inf, np.inf],
            )
        else:
            # Transmon limit: the noise-sensitivity term vanishes identically,
            # so Gamma_Omega and sigma_epsilon cannot be constrained and are
            # held fixed at zero instead of being fitted.
            p0 = [y0_guess, amplitude_guess, rabi_target, 0.0, 0.0]
            bounds = (
                [-2, -2, 0, 0, 0],
                [2, 2, 5 * max(rabi_target, 1), 1e-30, 1e-30],
            )

        try:
            popt, pcov = curve_fit(
                model,
                tau,
                prob,
                p0=p0,
                bounds=bounds,
                sigma=error,
                maxfev=200000,
            )
            perr = np.sqrt(np.diag(pcov))

            fitted_parameters[qubit] = popt.tolist()
            fitted_rabi_frequency[qubit] = popt[2]
            gamma_omega[qubit] = popt[3] if eta > 0 else np.nan
            gamma_omega_error[qubit] = perr[3] if eta > 0 else np.nan
            sigma_epsilon[qubit] = popt[4] if eta > 0 else np.nan
            sigma_epsilon_error[qubit] = perr[4] if eta > 0 else np.nan
            chi2[qubit] = [
                chi2_reduced(prob, model(tau, *popt), error),
                np.sqrt(2 / len(prob)),
            ]
        except Exception as e:
            log.warning(f"Rabi-coherence fit failed for qubit {qubit} due to {e}.")
            fitted_parameters[qubit] = [np.nan] * 5
            fitted_rabi_frequency[qubit] = np.nan
            gamma_omega[qubit] = np.nan
            gamma_omega_error[qubit] = np.nan
            sigma_epsilon[qubit] = np.nan
            sigma_epsilon_error[qubit] = np.nan
            chi2[qubit] = [np.nan, np.nan]

    return RabiCoherenceResults(
        gamma_omega=gamma_omega,
        gamma_omega_error=gamma_omega_error,
        sigma_epsilon=sigma_epsilon,
        sigma_epsilon_error=sigma_epsilon_error,
        fitted_rabi_frequency=fitted_rabi_frequency,
        fitted_parameters=fitted_parameters,
        chi2=chi2,
    )


def _plot(data: RabiCoherenceData, target: QubitId, fit: RabiCoherenceResults = None):
    """Plotting for the rabi-coherence experiment."""

    figures = []
    fitting_report = ""

    delays = data.delays(target)
    qubit_data = data[target]
    prob = qubit_data.prob
    error = qubit_data.error

    fig = go.Figure(
        [
            go.Scatter(
                x=delays,
                y=prob,
                mode="markers",
                name="Probability of 1",
                error_y=dict(array=error),
            ),
        ]
    )

    if fit is not None:
        delay_range = np.linspace(delays.min(), delays.max(), 5 * len(delays))
        popt = fit.fitted_parameters[target]
        if not np.any(np.isnan(popt)):
            gamma1 = data.gamma1[target]
            cos_theta = data.cos_theta[target]
            eta = data.sensitivity(target)
            fig.add_trace(
                go.Scatter(
                    x=delay_range,
                    y=_model(
                        delay_range * NS_TO_S,
                        *popt,
                        gamma1=gamma1,
                        eta=eta,
                        cos_theta=cos_theta,
                    ),
                    mode="lines",
                    name="Fit",
                    line=dict(dash="dot"),
                )
            )

            labels = [
                "Fitted Rabi frequency [Hz]",
                "Γ_Ω [Hz]",
                "σ_ε [Hz]",
                "χ2 reduced",
            ]
            values = [
                fit.fitted_rabi_frequency[target],
                fit.gamma_omega[target],
                fit.sigma_epsilon[target],
                fit.chi2[target],
            ]
            fitting_report = table_html(
                table_dict(target, labels, values, display_error=False)
            )

    fig.update_layout(
        xaxis_title="Interaction time τ [ns]",
        yaxis_title="Probability of state 1",
        title="Driven-evolution coherence decay",
    )
    figures.append(fig)

    return figures, fitting_report


rabi_coherence = Routine(_acquisition, _fit, _plot)
"""RabiCoherence Routine object."""
