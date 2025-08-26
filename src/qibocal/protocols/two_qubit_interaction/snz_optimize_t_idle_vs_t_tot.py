from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper
from qibolab._core.pulses.envelope import Snz

from qibocal.auto.operation import Parameters, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform

from .snz_optimize import SNZFinetuningData, SNZFinetuningResults
from .utils import fit_virtualz, order_pair
from .virtual_z_phases import create_sequence


@dataclass
class SNZDurationParamteters(Parameters):
    duration_min: float
    """Duration minimum."""
    duration_max: float
    """Duration maximum."""
    duration_step: float
    """Duration step."""
    t_idle_min: float
    """t_idle minimum."""
    t_idle_max: float
    """t_idle maximum."""
    t_idle_step: float
    """t_idle step."""
    theta_start: float
    """Virtual phase start angle."""
    theta_end: float
    """Virtual phase end angle."""
    theta_step: float
    """Virtual phase stop angle."""
    b_amplitude: float
    """SNZ B amplitude."""


@dataclass
class SNZDurationResults(SNZFinetuningResults): ...


OptimizeTwoQubitGateType = np.dtype(
    [
        ("theta", np.float64),
        ("target", np.float64),
        ("control", np.float64),
    ]
)


@dataclass
class SNZDurationData(SNZFinetuningData):
    durations: list = field(default_factory=list)
    """"Duratoins swept."""
    t_idles: list[float] = field(default_factory=list)
    """t_idles swept."""

    def register_qubit(
        self, target, control, setup, t_idle, theta, duration, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta)
        for i, duration in enumerate(duration):
            ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
            ar["theta"] = theta
            ar["control"] = prob_control[i]
            ar["target"] = prob_target[i]
            self.data[target, control, setup, t_idle, duration] = np.rec.array(ar)


def _aquisition(
    params: SNZDurationParamteters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZDurationData:
    t_idle_range = np.arange(
        params.t_idle_min, params.t_idle_max, params.t_idle_step, dtype=float
    )
    data = SNZDurationData()
    data.t_idles = t_idle_range.tolist()
    data.durations = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    ).tolist()
    data.angles = np.arange(
        params.theta_start, params.theta_end, params.theta_step
    ).tolist()
    for pair in targets:
        ordered_pair = order_pair(pair, platform)
        flux_channel = platform.qubits[ordered_pair[1]].flux
        target_vz = pair[0]
        other_qubit_vz = pair[1]
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

                sweeper_duration = Sweeper(
                    parameter=Parameter.duration,
                    range=(
                        params.duration_min,
                        params.duration_max,
                        params.duration_step,
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
                    [[sweeper_duration], [sweeper_theta]],
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
                    data.angles,
                    sweeper_duration.values,
                    results[ro_control.id],
                    results[ro_target.id],
                )

    return data


def _fit(
    data: SNZDurationData,
) -> SNZDurationResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    for pair in pairs:
        for duration in data.durations:
            for t_idle in data.t_idles:
                data_duration = data.filter_data_key(pair[0], pair[1], t_idle, duration)
                new_fitted_parameter, new_phases, new_angle, new_leak = fit_virtualz(
                    data_duration,
                    pair,
                    thetas=data.angles,
                    gate_repetition=1,
                    key=(pair[0], pair[1], t_idle, duration),
                )
                fitted_parameters |= new_fitted_parameter
                virtual_phases |= new_phases
                angles |= new_angle
                leakages |= new_leak

    results = SNZDurationResults(
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        angles=angles,
    )
    return results


def _plot(
    data: SNZDurationData,
    fit: SNZDurationResults,
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
        durations = []
        leakage = []
        target_q = target[0]
        control_q = target[1]

        for i in data.durations:
            for j in data.t_idles:
                t_idle.append(j)
                durations.append(i)
                cz.append(fit.angles[target_q, control_q, j, i])
                leakage.append(fit.leakages[target_q, control_q, j, i])

        fig.add_trace(
            go.Heatmap(
                x=durations,
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
                x=durations,
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
            xaxis1_title="Pulse duration [ns]",
            xaxis2_title="Pulse duration [ns]",
            yaxis1_title="t_idle [# samplings]",
            yaxis2_title="t_idle [# samplings]",
        )

    return [fig], fitting_report


snz_optimize_t_idle_vs_t_tot = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
