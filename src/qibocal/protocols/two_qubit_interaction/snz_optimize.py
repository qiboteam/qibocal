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
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .utils import order_pair
from .virtual_z_phases import create_sequence, fit_sinusoid, phase_diff


@dataclass
class SNZFinetuningParamteters(Parameters):
    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    amp_ratio_min: float
    """Amplitude minimum."""
    amp_ratio_max: float
    """Amplitude maximum."""
    amp_ratio_step: float
    """Amplitude step."""
    theta_start: float
    theta_end: float
    theta_step: float
    t_idling: float


@dataclass
class SNZFinetuningResults(Results):
    leakages: dict
    virtual_phases: dict
    fitted_parameters: dict
    angles: dict

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.

        Additional  manipulations required because of the Results class.
        """
        pairs = {
            (target, control) for target, control, _, _, _ in self.fitted_parameters
        }
        return key in pairs


OptimizeTwoQubitGateType = np.dtype(
    [
        ("amp", np.float64),
        ("theta", np.float64),
        ("rel_amplitude", np.float64),
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
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """"Amplitudes swept."""
    rel_amplitudes: list[float] = field(default_factory=list)
    """Durations swept."""
    angles: dict = field(default_factory=dict)

    def __getitem__(self, pair):
        """Extract data for pair."""
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, target, control, setup, ratio, theta, amp, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta) * len(amp)
        amplitude, angle = np.meshgrid(amp, theta, indexing="ij")
        ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
        ar["theta"] = angle.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[target, control, setup, ratio] = np.rec.array(ar)


def _aquisition(
    params: SNZFinetuningParamteters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZFinetuningData:
    ratio_range = np.arange(
        params.amp_ratio_min, params.amp_ratio_max, params.amp_ratio_step
    )
    data = SNZFinetuningData()
    data.rel_amplitudes = ratio_range.tolist()
    data.angles = np.arange(
        params.theta_start, params.theta_end, params.theta_step
    ).tolist()
    print(ratio_range)
    for pair in targets:
        ordered_pair = order_pair(pair, platform)
        flux_channel = platform.qubits[ordered_pair[1]].flux
        target_vz = pair[0]
        other_qubit_vz = pair[1]
        # Find CZ flux pulse
        cz_sequence = getattr(platform.natives.two_qubit[ordered_pair], "CZ")()
        flux_channel = platform.qubits[ordered_pair[1]].flux

        for cz_pulse in cz_sequence:
            if cz_pulse[0] == flux_channel:
                flux_pulse = cz_pulse[1]

        for ratio in ratio_range:
            for setup in ("I", "X"):
                flux_pulse = [
                    (
                        flux_channel,
                        Pulse(
                            amplitude=flux_pulse.amplitude,
                            duration=flux_pulse.duration,
                            envelope=Snz(
                                t_idling=params.t_idling,
                                b_amplitude=ratio,
                            ),
                        ),
                    )
                ]
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
                    flux_pulses=flux_pulse,
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
                # print("SEQUENCE")
                # for s in sequence:
                #     print(s)
                results = platform.execute(
                    [sequence],
                    [[sweeper_amplitude], [sweeper_theta]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                )

                # TODO: move this outside loops
                data.amplitudes[pair] = sweeper_amplitude.values.tolist()
                # print(data.thetas)
                data.thetas = -1 * sweeper_theta.values
                data.thetas = data.thetas.tolist()
                # print(data.thetas)
                # data.durations[ordered_pair] = sweeper_duration.values.tolist()
                data.register_qubit(
                    target_vz,
                    other_qubit_vz,
                    setup,
                    ratio,
                    data.thetas,
                    sweeper_amplitude.values,
                    results[ro_control.id],
                    results[ro_target.id],
                )

                # print(results[ro_target.id])
                # plt.plot(
                #     data.thetas,
                #     results[ro_target.id].ravel(),
                #     label=f"{target_vz} {other_qubit_vz} {setup}",
                # )
                # plt.xlabel("Theta")
                # plt.ylabel("Probability")
                # plt.title(f"Pair {target_vz} {other_qubit_vz} {setup}")
                # plt.legend()
                # plt.savefig(
                #     f"pair_{target_vz}_{other_qubit_vz}_{setup}.png", dpi=300, bbox_inches="tight"
                # )
                # # print(data)
    # print(data)
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
        # TODO: improve this
        # ord_pair = next(iter(data.amplitudes))[:2]
        # for rel_amplitude in data.rel_amplitudes:
        for amplitude in data.amplitudes[pair]:
            # virtual_phases[ord_pair[0], ord_pair[1], amplitude, rel_amplitude] = {}
            # leakages[ord_pair[0], ord_pair[1], amplitude, rel_amplitude] = {}
            # breakpoint()
            for target, control, setup, rel_amplitude in data[pair]:
                selected_data = data[pair][target, control, setup, rel_amplitude]
                target_data = selected_data.prob_target[selected_data.amp == amplitude,]
                # breakpoint()
                try:
                    params = fit_sinusoid(
                        np.array(data.thetas), target_data, gate_repetition=1
                    )
                    fitted_parameters[
                        target, control, setup, amplitude, rel_amplitude
                    ] = params
                except Exception as e:
                    log.warning(f"Fit failed for pair ({target, control}) due to {e}.")

            for target, control, setup, rel_amplitude in data[pair]:
                if setup == "I":  # The loop is the same for setup I or X
                    # try:
                    angles[target, control, amplitude, rel_amplitude] = phase_diff(
                        fitted_parameters[
                            target, control, "X", amplitude, rel_amplitude
                        ][2],
                        fitted_parameters[
                            target, control, "I", amplitude, rel_amplitude
                        ][2],
                    )
                    virtual_phases[target, control, amplitude, rel_amplitude] = (
                        fitted_parameters[
                            target, control, "I", amplitude, rel_amplitude
                        ][2]
                    )

                    # leakage estimate: L = m /2
                    # See NZ paper from Di Carlo
                    # approximation which does not need qutrits
                    # https://arxiv.org/pdf/1903.02492.pdf
                    data_x = data[pair][target, control, "X", rel_amplitude]
                    data_i = data[pair][target, control, "I", rel_amplitude]
                    leakages[target, control, amplitude, rel_amplitude] = 0.5 * np.mean(
                        data_x[data_x.amp == amplitude].prob_control
                        - data_i[data_i.amp == amplitude].prob_control
                    )

                    # except KeyError:
                    #     pass
    results = SNZFinetuningResults(
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        angles=angles,
    )
    # print(results)
    return results


def _plot(
    data: SNZFinetuningData,
    fit: SNZFinetuningResults,
    target: QubitPairId,
):
    """Plot routine for OptimizeTwoQubitGate."""
    fitting_report = ""
    qubits = next(iter(data.amplitudes))[:2]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Angle",
            "Leakage",
        ),
    )
    # print("FFFFFFF", fit)
    # print(data)
    if fit is not None:
        # print(data.amplitudes)
        # print(target)
        cz = []
        rel_amp = []
        amps = []
        leakage = []
        target_q = target[0]
        control_q = target[1]

        for i in data.amplitudes[
            target
        ]:  # TODO: solve asymettry amplitude/ rel_amplitude
            for j in data.rel_amplitudes:
                rel_amp.append(j)
                amps.append(i)
                cz.append(fit.angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[qubits[0], qubits[1], i, j])

        # condition = [target_q, control_q] == list(target)
        fig.add_trace(
            go.Heatmap(
                x=amps,
                y=rel_amp,
                z=cz,
                zmin=0,
                zmax=2 * np.pi,
                name="{fit.native} angle",
                colorbar_x=-0.1,
                colorscale="Twilight",
                # showscale=condition,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=amps,
                y=rel_amp,
                z=leakage,
                name="Leakage",
                # showscale=condition,
                colorscale="Inferno",
                zmin=0,
                zmax=0.2,
            ),
            row=1,
            col=2,
        )
        # fitting_report = table_html(
        #     table_dict(
        #         [qubits[1], qubits[1]],
        #         [
        #             "Flux pulse amplitude [a.u.]",
        #             "Flux pulse duration [ns]",
        #         ],
        #         [
        #             np.round(fit.best_amp[qubits], 4),
        #             np.round(fit.best_dur[qubits], 4),
        #         ],
        #     )
        # )

        fig.update_layout(
            xaxis1_title="Amplitude A [a.u.]",
            xaxis2_title="Amplitude A [a.u.]",
            yaxis1_title="Rel. Amp. B/A [a.u.]",
            yaxis2_title="Rel. Amp. B/A [a.u.]",
        )

    return [fig], fitting_report


snz_optimize = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
