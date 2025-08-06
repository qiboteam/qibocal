from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper
from qibolab._core.pulses.envelope import Snz

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from .utils import fit_virtualz, order_pair
from .virtual_z_phases import create_sequence


@dataclass
class SNZFinetuningParamteters(Parameters):
    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    amp_ratio_min: float
    """Minimum amplitude ratio of the two SNZ amplitudes"""
    amp_ratio_max: float
    """Maximum amplitude ratio of the two SNZ amplitudes"""
    amp_ratio_step: float
    """Amplitude ratio step of the two SNZ amplitudes"""
    theta_start: float
    """Virtual phase start angle."""
    theta_end: float
    """Virtual phase end angle."""
    theta_step: float
    """Virtual phase stop angle."""
    t_idling: float
    """SNZ idling time, in number of steps."""
    flux_time_delay: float = 0
    """Wait time after flux pulse."""


@dataclass
class SNZFinetuningResults(Results):
    leakages: dict
    """Qubit leakage."""
    virtual_phases: dict
    """Virtual Z phase correction."""
    fitted_parameters: dict
    """Fit parameters."""
    angles: dict
    """Native SNZ angle."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.

        Additional  manipulations required because of the Results class.
        """
        return key in self.pairs

    @property
    def pairs(self):
        pairs = {(target, control) for target, control, _, _ in self.angles}
        return pairs


OptimizeTwoQubitGateType = np.dtype(
    [
        ("theta", np.float64),
        ("target", np.float64),
        ("control", np.float64),
    ]
)


@dataclass
class SNZFinetuningData(Data):
    data: dict[tuple, npt.NDArray[OptimizeTwoQubitGateType]] = field(
        default_factory=dict
    )
    """Raw data."""
    amplitudes: list[float] = field(default_factory=list)
    """"Amplitudes swept."""
    rel_amplitudes: list[float] = field(default_factory=list)
    """Durations swept."""
    angles: list = field(default_factory=list)
    """Virtual phases."""

    def __getitem__(self, pair):
        """Extract data for pair."""
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def filter_data_key(self, target, control, rel_amplitude, amplitude):
        filtered_data = {}
        for index, value in self.data.items():
            new_index = list(index)
            setup = new_index.pop(2)
            if new_index == [target, control, rel_amplitude, amplitude]:
                filtered_data[target, control, setup] = value
        return filtered_data

    @property
    def order_pairs(self):
        pairs = []
        for key in self.data.keys():
            pairs.append([key[0], key[1]])
        return np.unique(pairs, axis=0).tolist()

    @property
    def swept_virtual_phases(self):
        """List of swept phases."""
        return [-1 * i for i in self.angles]

    def register_qubit(
        self, target, control, setup, ratio, theta, amp, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta)
        for i, amplitude in enumerate(amp):
            ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
            ar["theta"] = theta
            ar["control"] = prob_control[i]
            ar["target"] = prob_target[i]
            self.data[target, control, setup, ratio, amplitude] = np.rec.array(ar)


def _aquisition(
    params: SNZFinetuningParamteters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SNZFinetuningData:
    """Acquisition for the optimization of SNZ amplitudes. The amplitude of the
    SNZ pulse and its amplitude ratio (B/A) are swept while the virtual phase correction
    experiment is performed.
    """
    ratio_range = np.arange(
        params.amp_ratio_min, params.amp_ratio_max, params.amp_ratio_step
    )
    data = SNZFinetuningData()
    data.rel_amplitudes = ratio_range.tolist()
    data.amplitudes = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
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
        cz_sequence = platform.natives.two_qubit[ordered_pair].CZ
        flux_channel = platform.qubits[ordered_pair[1]].flux
        flux_pulses = list(cz_sequence.channel(flux_channel))
        assert len(flux_pulses) == 1, "Only 1 flux pulse is supported"
        flux_pulse = flux_pulses[0]

        for ratio in ratio_range:
            for setup in ("I", "X"):
                flux_pulse = Pulse(
                    amplitude=flux_pulse.amplitude,
                    duration=flux_pulse.duration,
                    envelope=Snz(
                        t_idling=params.t_idling,
                        b_amplitude=ratio,
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
                    dt=params.flux_time_delay,  # TODO: when dt is zero a 16 ns is added
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
                    ratio,
                    data.swept_virtual_phases,
                    sweeper_amplitude.values,
                    results[ro_control.id],
                    results[ro_target.id],
                )

    return data


def _fit(
    data: SNZFinetuningData,
) -> SNZFinetuningResults:
    """Repetition of correct virtual phase fit for all configurations."""

    fitted_parameters = {}
    pairs = data.order_pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    for pair in pairs:
        for amplitude in data.amplitudes:
            for rel_amplitude in data.rel_amplitudes:
                data_amplitude = data.filter_data_key(
                    pair[0], pair[1], rel_amplitude, amplitude
                )
                new_fitted_parameter, new_phases, new_angle, new_leak = fit_virtualz(
                    data_amplitude,
                    pair,
                    thetas=data.angles,
                    gate_repetition=1,
                    key=(pair[0], pair[1], amplitude, rel_amplitude),
                )
                fitted_parameters |= new_fitted_parameter
                virtual_phases |= new_phases
                angles |= new_angle
                leakages |= new_leak

    return SNZFinetuningResults(
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        angles=angles,
    )


def _plot(
    data: SNZFinetuningData,
    fit: SNZFinetuningResults,
    target: QubitPairId,
):
    """Plot routine for SNZ optimization."""
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
        rel_amp = []
        amps = []
        leakage = []
        target_q = target[0]
        control_q = target[1]
        for i in data.amplitudes:
            for j in data.rel_amplitudes:
                rel_amp.append(j)
                amps.append(i)
                cz.append(fit.angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[target_q, control_q, i, j])

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
            yaxis1_title="Rel. Amp. B/A [a.u.]",
            yaxis2_title="Rel. Amp. B/A [a.u.]",
        )

    return [fig], fitting_report


snz_optimize = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
