from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

from .utils import fit_snz_optimize, order_pair
from .virtual_z_phases import create_sequence


@dataclass
class SNZFinetuningParamteters(Parameters):
    # TODO: amplitude_min should be a list to be able to execute on multiple pairs.
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
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
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

    @property
    def swept_virtual_phases(self):
        """List of swept phases."""
        return [-1 * i for i in self.angles]

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
    """Acquisition for the optimization of SNZ amplitudes. The amplitude of the
    SNZ pulse and its amplitude ratio (B/A) are swept while the virtual phase correction
    experiment is performed.
    """
    ratio_range = np.arange(
        params.amp_ratio_min, params.amp_ratio_max, params.amp_ratio_step
    )
    data = SNZFinetuningData()
    data.rel_amplitudes = ratio_range.tolist()
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

                # TODO: move this outside loops
                data.amplitudes[pair] = sweeper_amplitude.values.tolist()
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

    virtual_phases, fitted_parameters, leakages, angles = fit_snz_optimize(data)
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
    qubits = next(iter(data.amplitudes))[:2]
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

        for i in data.amplitudes[
            target
        ]:  # TODO: solve asymettry amplitude/ rel_amplitude
            for j in data.rel_amplitudes:
                rel_amp.append(j)
                amps.append(i)
                cz.append(fit.angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[qubits[0], qubits[1], i, j])

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
