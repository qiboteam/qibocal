from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper
from qibolab._core.pulses.envelope import Snz

from qibocal.auto.operation import Data, Parameters, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform

from .snz_optimize import (
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
    tp: float
    """Gate time."""
    theta_start: float
    """Virtual phase start angle."""
    theta_end: float
    """Virtual phase end angle."""
    theta_step: float
    """Virtual phase stop angle."""
    b_amplitude: float
    """SNZ B amplitude."""

    @property
    def theta_range(self) -> np.ndarray:
        return np.arange(self.theta_start, self.theta_end, self.theta_step)

    @property
    def amplitude_range(self) -> np.ndarray:
        return np.arange(
            self.amplitude_min,
            self.amplitude_max,
            self.amplitude_step,
        )

    @property
    def t_idle_range(self) -> np.ndarray:
        return np.arange(self.t_idle_min, self.t_idle_max, self.t_idle_step)


@dataclass
class SNZIdlingResults(SNZFinetuningResults): ...


@dataclass
class SNZIdlingData(Data):
    t_idles: list[float] = field(default_factory=list)
    _sorted_pairs: list[QubitPairId] = field(default_factory=dict)
    thetas: list = field(default_factory=list)
    """Angles swept."""
    amplitudes: list = field(default_factory=list)
    """"Amplitudes swept."""
    t_idles: list = field(default_factory=list)
    """Durations swept."""
    data: dict[tuple, npt.NDArray] = field(default_factory=dict)

    @property
    def sorted_pairs(self):
        return [
            pair if isinstance(pair, tuple) else tuple(pair)
            for pair in self._sorted_pairs
        ]

    @sorted_pairs.setter
    def sorted_pairs(self, value):
        self._sorted_pairs = value

    def parse(self, i, j):
        return {
            key: value[
                :,
                i,
                j,
            ]
            for key, value in self.data.items()
        }


def _aquisition(
    params: SNZIdlingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZIdlingData:
    """Acquisition for the optimization of SNZ amplitudes. The amplitude of the
    SNZ pulse and its idling time are swept while the virtual phase correction
    experiment is performed.
    """
    data = SNZIdlingData(
        _sorted_pairs=[order_pair(pair, platform) for pair in targets],
        thetas=params.theta_range.tolist(),
        amplitudes=params.amplitude_range.tolist(),
        t_idles=params.t_idle_range.tolist(),
    )
    for ordered_pair in data.sorted_pairs:
        flux_channel = platform.qubits[ordered_pair[1]].flux
        target_vz = ordered_pair[0]
        other_qubit_vz = ordered_pair[1]

        # Find CZ flux pulse
        cz_sequence = getattr(platform.natives.two_qubit[ordered_pair], "CZ")()
        flux_channel = platform.qubits[ordered_pair[1]].flux
        flux_pulses = list(cz_sequence.channel(flux_channel))
        assert len(flux_pulses) == 1, "Only 1 flux pulse is supported"
        flux_pulse = flux_pulses[0]

        data.data[target_vz, other_qubit_vz, "I"] = []
        data.data[target_vz, other_qubit_vz, "X"] = []
        for t_idle in params.t_idle_range:
            for setup in ("I", "X"):
                flux_pulse = Pulse(
                    amplitude=flux_pulse.amplitude,
                    duration=params.tp + t_idle,
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

                data.data[target_vz, other_qubit_vz, setup].append(
                    np.stack([results[ro_target.id], results[ro_control.id]])
                )

        for setup in ("I", "X"):
            data.data[target_vz, other_qubit_vz, setup] = np.moveaxis(
                np.array(data.data[target_vz, other_qubit_vz, setup]), [0, 1], [2, 0]
            )

    return data


def _fit(
    data: SNZIdlingData,
) -> SNZIdlingResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    virtual_phases = {}
    angles = {}
    leakages = {}

    for pair in data.sorted_pairs:
        _pair = tuple(pair)
        angles[_pair], leakages[_pair], virtual_phases[_pair] = [], [], []
        (
            fitted_parameters[_pair[0], _pair[1], "I"],
            fitted_parameters[_pair[0], _pair[1], "X"],
        ) = [], []

        for i in range(len(data.amplitudes)):
            for j in range(len(data.t_idles)):
                # if i == 0 and j == 0:
                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.scatter(data.thetas, data.parse(j, i)[_pair[0], _pair[1], "X"][0], label="Target X")
                #     plt.scatter(data.thetas, data.parse(j, i)[_pair[0], _pair[1], "I"][0], label="Target I")
                #     plt.scatter(data.thetas, data.parse(j, i)[_pair[0], _pair[1], "X"][1], label="Control X")
                #     plt.scatter(data.thetas, data.parse(j, i)[_pair[0], _pair[1], "I"][1], label="Control I")
                #     plt.legend()
                #     plt.savefig("test.png")
                new_fitted_parameter, new_phases, new_angle, new_leak = fit_virtualz(
                    data.parse(i, j),
                    _pair,
                    thetas=data.thetas,
                    gate_repetition=1,
                )
                angles[_pair].append(new_angle[_pair])
                leakages[_pair].append(new_leak[_pair])
                virtual_phases[_pair].append(new_phases[_pair])
                for setup in ["I", "X"]:
                    fitted_parameters[_pair[0], _pair[1], setup].append(
                        new_fitted_parameter[_pair, setup]
                    )
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
    if target not in data.sorted_pairs:
        target = (target[1], target[0])
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Angle",
            "Leakage",
        ),
    )
    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=data.amplitudes,
                y=data.t_idles,
                z=np.array(fit.angles[target])
                .reshape(len(data.amplitudes), len(data.t_idles))
                .T,
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
                x=data.amplitudes,
                y=data.t_idles,
                z=np.array(fit.leakages[target])
                .reshape(len(data.amplitudes), len(data.t_idles))
                .T,
                name="Leakage",
                colorscale="Inferno",
                zmin=0,
                zmax=0.25,
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


# TODO: Add update function
snz_optimize_t_idle = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
