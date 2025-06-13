from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import plotly.colors.cyclical
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper
from qibolab._core.pulses.envelope import Snz

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .snz_optimize import SNZFinetuningResults
from .utils import fit_sinusoid, order_pair, phase_diff
from .virtual_z_phases import create_sequence


@dataclass
class SNZFinetuningParamteters(Parameters):
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
class SNZFinetuningResults(SNZFinetuningResults): ...


OptimizeTwoQubitGateType = np.dtype(
    [
        ("duration", np.float64),
        ("theta", np.float64),
        ("t_idle", np.float64),
        ("prob_target", np.float64),
        ("prob_control", np.float64),
    ]
)


@dataclass
class SNZFinetuningData(Data):
    data: dict[tuple, npt.NDArray[OptimizeTwoQubitGateType]] = field(
        default_factory=dict
    )
    """Raw data."""
    thetas: list = field(default_factory=list)
    """Angles swept."""
    durations: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """"Duratoins swept."""
    t_idles: list[float] = field(default_factory=list)
    """t_idles swept."""
    angles: dict = field(default_factory=dict)

    def __getitem__(self, pair):
        """Extract data for pair."""
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, target, control, setup, t_idle, theta, duration, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta) * len(duration)
        duration, angle = np.meshgrid(duration, theta, indexing="ij")
        ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
        ar["theta"] = angle.ravel()
        ar["duration"] = duration.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[target, control, setup, t_idle] = np.rec.array(ar)


def _aquisition(
    params: SNZFinetuningParamteters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZFinetuningData:
    t_idle_range = np.arange(
        params.t_idle_min, params.t_idle_max, params.t_idle_step, dtype=float
    )
    data = SNZFinetuningData()
    data.t_idles = t_idle_range.tolist()
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

                # TODO: move this outside loops
                data.durations[pair] = sweeper_duration.values.tolist()
                data.thetas = -1 * sweeper_theta.values
                data.thetas = data.thetas.tolist()
                data.register_qubit(
                    target_vz,
                    other_qubit_vz,
                    setup,
                    t_idle,
                    data.thetas,
                    sweeper_duration.values,
                    results[ro_control.id],
                    results[ro_target.id],
                )

    return data


def _fit(
    data: SNZFinetuningData,
) -> SNZFinetuningResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    # FIXME: experiment should be for single pair
    for pair in pairs:
        for duration in data.durations[pair]:
            for target, control, setup, t_idle in data[pair]:
                selected_data = data[pair][target, control, setup, t_idle]
                target_data = selected_data.prob_target[
                    selected_data.duration == duration,
                ]
                try:
                    params = fit_sinusoid(
                        np.array(data.thetas), target_data, gate_repetition=1
                    )
                    fitted_parameters[target, control, setup, duration, t_idle] = params
                except Exception as e:
                    log.warning(f"Fit failed for pair ({target, control}) due to {e}.")
            try:
                for target, control, setup, t_idle in data[pair]:
                    if setup == "I":  # The loop is the same for setup I or X
                        angles[target, control, duration, t_idle] = phase_diff(
                            fitted_parameters[target, control, "X", duration, t_idle][
                                2
                            ],
                            fitted_parameters[target, control, "I", duration, t_idle][
                                2
                            ],
                        )
                        virtual_phases[target, control, duration, t_idle] = (
                            fitted_parameters[target, control, "I", duration, t_idle][2]
                        )

                        # leakage estimate: L = m /2
                        # See NZ paper from Di Carlo
                        # approximation which does not need qutrits
                        # https://arxiv.org/pdf/1903.02492.pdf
                        data_x = data[pair][target, control, "X", t_idle]
                        data_i = data[pair][target, control, "I", t_idle]
                        leakages[target, control, duration, t_idle] = 0.5 * np.mean(
                            data_x[data_x.duration == duration].prob_control
                            - data_i[data_i.duration == duration].prob_control
                        )
            except KeyError:
                pass

    results = SNZFinetuningResults(
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        angles=angles,
    )
    return results


def _plot(
    data: SNZFinetuningData,
    fit: SNZFinetuningResults,
    target: QubitPairId,
):
    """Plot routine for OptimizeTwoQubitGate."""
    fitting_report = ""
    qubits = next(iter(data.durations))[:2]
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

        for i in data.durations[target]:
            for j in data.t_idles:
                t_idle.append(j)
                durations.append(i)
                cz.append(fit.angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[qubits[0], qubits[1], i, j])

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
