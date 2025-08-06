from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper
from qibolab._core.pulses.envelope import Snz

from qibocal.auto.operation import Parameters, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform

from .snz_optimize import (
    OptimizeTwoQubitGateType,
    SNZFinetuningData,
    SNZFinetuningResults,
)
from .utils import fit_virtualz, order_pair
from .virtual_z_phases import create_sequence


@dataclass
class SNZIdlingParameters(Parameters):
    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    t_idle_min: float
    """Amplitude minimum."""
    t_idle_max: float
    """Amplitude maximum."""
    t_idle_step: float
    """Amplitude step."""
    theta_start: float
    """Virtual phase start angle."""
    theta_end: float
    """Virtual phase end angle."""
    theta_step: float
    """Virtual phase stop angle."""
    b_amplitude: float
    """SNZ B amplitude."""


@dataclass
class SNZIdlingResults(SNZFinetuningResults): ...


@dataclass
class SNZIdlingData(SNZFinetuningData):
    t_idles: list[float] = field(default_factory=list)

    def register_qubit(
        self, target, control, setup, t_idle, theta, amp, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta)
        for i, amplitude in enumerate(amp):
            ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
            ar["theta"] = theta
            ar["control"] = prob_control[i]
            ar["target"] = prob_target[i]
            self.data[target, control, setup, t_idle, amplitude] = np.rec.array(ar)


def _aquisition(
    params: SNZIdlingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZIdlingData:
    """Acquisition for the optimization of SNZ amplitudes. The amplitude of the
    SNZ pulse and its idling time are swept while the virtual phase correction
    experiment is performed.
    """
    t_idle_range = np.arange(
        params.t_idle_min, params.t_idle_max, params.t_idle_step, dtype=float
    )
    data = SNZIdlingData()
    data.t_idles = t_idle_range.tolist()
    data.angles = np.arange(
        params.theta_start, params.theta_end, params.theta_step
    ).tolist()
    for pair in targets:
        ordered_pair = order_pair(pair, platform)
        flux_channel = platform.qubits[ordered_pair[1]].flux
        target_vz = pair[0]
        other_qubit_vz = pair[1]

        data.amplitudes = np.arange(
            params.amplitude_min, params.amplitude_max, params.amplitude_step
        ).tolist()
        # Find CZ flux pulse
        cz_sequence = getattr(platform.natives.two_qubit[ordered_pair], "CZ")()
        flux_channel = platform.qubits[ordered_pair[1]].flux
        flux_pulses = list(cz_sequence.channel(flux_channel))
        assert len(flux_pulses) == 1, "Only 1 flux pulse is supported"
        flux_pulse = flux_pulses[0]

        for t_idle in t_idle_range:
            for setup in ("I", "X"):
                flux_pulse = Pulse(
                    amplitude=flux_pulse.amplitude,
                    duration=flux_pulse.duration,
                    envelope=Snz(
                        t_idling=t_idle,
                        b_amplitude=params.b_amplitude,
                    ),
                )
                (
                    sequence,
                    flux_pulse,
                    theta_pulse,
                ) = create_sequence(
                    platform,
                    setup,
                    target_vz,
                    other_qubit_vz,
                    ordered_pair,
                    "CZ",
                    dt=0,
                    flux_pulse=flux_pulse,
                )
                sweeper_theta = Sweeper(
                    parameter=Parameter.phase,
                    range=(-params.theta_start, -params.theta_end, -params.theta_step),
                    pulses=theta_pulse,
                )

                sweeper_amplitude = Sweeper(
                    parameter=Parameter.amplitude,
                    range=(
                        params.amplitude_min,
                        params.amplitude_max,
                        params.amplitude_step,
                    ),
                    pulses=[flux_pulse],
                )
                ro_target = list(
                    sequence.channel(platform.qubits[target_vz].acquisition)
                )[-1]
                ro_control = list(
                    sequence.channel(platform.qubits[other_qubit_vz].acquisition)
                )[-1]
                results = platform.execute(
                    [sequence],
                    [[sweeper_amplitude], [sweeper_theta]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                )

                data.register_qubit(
                    target_vz,
                    other_qubit_vz,
                    setup,
                    t_idle,
                    data.swept_virtual_phases,
                    sweeper_amplitude.values,
                    results[ro_control.id],
                    results[ro_target.id],
                )

    return data


def _fit(
    data: SNZIdlingData,
) -> SNZIdlingResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    pairs = data.order_pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    for pair in pairs:
        for amplitude in data.amplitudes:
            for t_idle in data.t_idles:
                data_amplitude = data.filter_data_key(
                    pair[0], pair[1], t_idle, amplitude
                )
                new_fitted_parameter, new_phases, new_angle, new_leak = fit_virtualz(
                    data_amplitude,
                    pair,
                    thetas=data.angles,
                    gate_repetition=1,
                    key=(pair[0], pair[1], amplitude, t_idle),
                )
                fitted_parameters |= new_fitted_parameter
                virtual_phases |= new_phases
                angles |= new_angle
                leakages |= new_leak

    return SNZIdlingResults(
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        angles=angles,
    )


def _plot(
    data: SNZIdlingData,
    fit: SNZIdlingResults,
    target: QubitPairId,
):
    """Plot routine for OptimizeTwoQubitGate."""
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Angle",
            "Leakage",
        ),
    )
    if fit is not None:
        cz = []
        t_idle = []
        amps = []
        leakage = []
        target_q = target[0]
        control_q = target[1]

        for i in data.amplitudes:
            for j in data.t_idles:
                t_idle.append(j)
                amps.append(i)
                cz.append(fit.angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[target_q, control_q, i, j])

        fig.add_trace(
            go.Heatmap(
                x=amps,
                y=t_idle,
                z=cz,
                zmin=0,
                zmax=2 * np.pi,
                name="{fit.native} angle",
                colorbar_x=-0.1,
                colorscale="Twilight",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=amps,
                y=t_idle,
                z=leakage,
                name="Leakage",
                colorscale="Inferno",
                zmin=0,
                zmax=0.2,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            xaxis1_title="Amplitude A [a.u.]",
            xaxis2_title="Amplitude A [a.u.]",
            yaxis1_title="t_idle [# samplings]",
            yaxis2_title="t_idle [# samplings]",
        )

    return [fig], fitting_report


snz_optimize_t_idle = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
