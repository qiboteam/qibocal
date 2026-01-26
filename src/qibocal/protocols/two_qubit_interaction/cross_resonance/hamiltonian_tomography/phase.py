"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from .....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from .....calibration import CalibrationPlatform
from ..utils import Basis, SetControl, cr_sequence
from .length import HamiltonianTomographyCRLengthResults
from .utils import (
    EPS,
    HamiltonianTerm,
    calibration_cr_plot,
    cyclic_prob,
    phase_tomography_cr_fit,
)

HamiltonianTomographyCRPhaseType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("phase", np.float64),
        ("x", np.int64),
    ]
)
"""Custom dtype for CR amplitude."""


@dataclass
class HamiltonianTomographyCRPhaseParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    cancellation_calibration: bool
    """Sweep over control or target qubit amplitude"""
    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""
    control_amplitude: float
    """Initial amplitude of CR pulse."""
    control_phase: float
    """Initial amplitude of CR pulse."""
    phase_end: float
    """Final amplitude of CR pulse."""
    phase_step: float
    """Step CR pulse amplitude."""
    phase: float = 0
    """Phase of CR pulse."""
    target_amplitude: float = 0
    """Amplitude of cancellation pulse."""
    target_phase: float = 0
    """Phase of target pulse."""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """
    phase_plot_dict: dict[QubitPairId, list] = field(default_factory=dict)
    """
    Dictionary containing the values of phases for which plot hamiltonian tomography
    for each qubit.
    """

    @property
    def phase_range(self) -> np.ndarray:
        """Amplitude range for CR pulses."""
        return np.arange(
            self.control_phase
            if not self.cancellation_calibration
            else self.target_phase,
            self.phase_end,
            self.phase_step,
        )

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRPhaseResults(Results):
    """HamiltonianTomographyCRAmplitude outputs."""

    cancellation_calibration: bool
    """Sweep over control or target qubit amplitude"""

    hamiltonian_terms: dict[QubitId, QubitId] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters from X,Y,Z expectation values for different phases."""

    tomography_length_parameters: HamiltonianTomographyCRLengthResults = field(
        default_factory=dict
    )
    """Fitted parameters from X,Y,Z expectation values."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRPhaseData(Data):
    """Data structure for CR Amplitude."""

    cancellation_calibration: bool
    phases: list = None
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRPhaseType],
    ] = field(default_factory=dict)
    """Raw data acquired."""
    phase_plot_dict: dict[QubitPairId, list] = field(default_factory=dict)
    """
    Dictionary containing the values of amplitude for which plot hamiltonian tomography
    for each qubit.
    """

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}

    def select_phase(self, phase: float):
        new_data = HamiltonianTomographyCRPhaseData(
            cancellation_calibration=self.cancellation_calibration
        )
        new_data.data = {k: d[d.phase == phase] for k, d in self.data.items()}
        return new_data

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit."""
        duration_list = data_dict["x"]
        phase_list = data_dict["phase"]
        size = len(duration_list) * len(phase_list)
        ar = np.empty(size, dtype=dtype)
        phases, durations = np.meshgrid(phase_list, duration_list)
        ar["x"] = durations.ravel()
        ar["phase"] = phases.ravel()
        ar["prob_target"] = data_dict["prob_target"].ravel()
        ar["error_target"] = data_dict["error_target"]
        ar["prob_control"] = data_dict["prob_control"].ravel()
        ar["error_control"] = data_dict["error_control"]

        self.data[data_keys] = np.rec.array(ar)


def _acquisition(
    params: HamiltonianTomographyCRPhaseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRPhaseData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRPhaseData(
        cancellation_calibration=params.cancellation_calibration,
        phases=params.phase_range.astype(float).tolist(),
        phase_plot_dict=params.phase_plot_dict,
    )

    for pair in targets:
        control, target = pair
        pair = (control, target)

        if params.cancellation_calibration:
            ctrl_phase = params.control_phase
            target_phase = params.phase_end
        else:
            ctrl_phase = params.phase_end
            target_phase = params.target_phase

        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    setup=setup,
                    amplitude=ctrl_phase,
                    phase=params.phase,
                    target_amplitude=params.target_amplitude,
                    target_phase=target_phase,
                    duration=params.pulse_duration_end,
                    echo=params.echo,
                    basis=basis,
                )
                if params.interpolated_sweeper:
                    length_sweeper = Sweeper(
                        parameter=Parameter.duration_interpolated,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses,
                    )
                else:
                    length_sweeper = Sweeper(
                        parameter=Parameter.duration,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses + delays,
                    )

                phase_sweeper = Sweeper(
                    parameter=Parameter.relative_phase,
                    values=params.phase_range,
                    pulses=(
                        cr_target_pulses
                        if params.cancellation_calibration
                        else cr_pulses
                    ),
                )

                updates = []
                updates.append(
                    {
                        platform.qubits[control].drive_extra[target]: {
                            "frequency": platform.config(
                                platform.qubits[target].drive
                            ).frequency
                        }
                    }
                )
                results = platform.execute(
                    [sequence],
                    [[length_sweeper], [phase_sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                    updates=updates,
                )
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id

                prob_target = cyclic_prob(results[target_acq_handle], state=1).ravel()
                prob_control = cyclic_prob(results[control_acq_handle], state=1).ravel()

                data.register_qubit(
                    HamiltonianTomographyCRPhaseType,
                    (control, target, basis, setup),
                    dict(
                        x=length_sweeper.values,
                        phase=phase_sweeper.values,
                        prob_target=1 - 2 * prob_target,
                        error_target=(
                            EPS
                            + 2
                            * np.sqrt(prob_target * (1 - prob_target) / params.nshots)
                        ).tolist(),
                        prob_control=prob_control,
                        error_control=(
                            EPS
                            + np.sqrt(prob_control * (1 - prob_control) / params.nshots)
                        ).tolist(),
                    ),
                )
    return data


def _fit(
    data: HamiltonianTomographyCRPhaseData,
) -> HamiltonianTomographyCRPhaseResults:
    """Post-processing function for HamiltonianTomographyCRAmplitude.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    length_tom_params, hamiltonian_terms, fitted_parameters = phase_tomography_cr_fit(
        data=data,
    )
    length_tom_params = HamiltonianTomographyCRLengthResults(
        fitted_parameters=length_tom_params
    )

    return HamiltonianTomographyCRPhaseResults(
        cancellation_calibration=data.cancellation_calibration,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        tomography_length_parameters=length_tom_params,
    )


def _plot(
    data: HamiltonianTomographyCRPhaseData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRPhaseResults,
):
    """Plotting function for HamiltonianTomographyCRAmplitude."""
    figs, fitting_report = calibration_cr_plot(data, target, fit)
    return figs, fitting_report


hamiltonian_tomography_cr_phase = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
