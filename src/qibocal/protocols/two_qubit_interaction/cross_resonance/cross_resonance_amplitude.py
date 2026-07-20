from dataclasses import dataclass, field

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    Sweeper,
)
from scipy.constants import kilo

from qibocal.auto.operation import (
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html

from .cr_parent_classes import (
    Basis,
    HamiltonianTerm,
    HamiltonianTomographyData,
    HamiltonianTomographyResults,
    HamiltonianTomographyType,
    SetControl,
)
from .cross_resonance_processing import (
    extract_hamiltonian_terms,
    tomography_cr_fit,
)
from .plotting import tomography_cr_plot
from .utils import (
    cross_resonance_experiment,
    update_cnot_from_fit,
)

__all__ = ["cr_amplitude"]


@dataclass(kw_only=True)
class HamiltonianTomographyCRAmplParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    pulse_duration: float
    """Duration of CR pulse [ns]."""
    amplitude_range: tuple[float, float, float]
    """Range of amplitudes for CR pulse (start, end, step)."""
    control_phase: float = 0.0
    """Phase of the CR pulse on the control qubit"""
    target_amplitude: float | None = None
    """Amplitude of the Cancellation pulse on the target qubit"""
    target_phase: float = 0.0
    """Phase of the Cancellation pulse on the target qubit"""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """


@dataclass(kw_only=True)
class HamiltonianTomographyCRAmplResults(HamiltonianTomographyResults):
    """HamiltonianTomographyCRAmpl outputs."""

    cr_duration: float
    control_phase: float
    target_amplitude: float | None
    target_phase: float
    cr_amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Estimated amplitudes of CR gate."""
    hamiltonian_terms: dict[tuple[QubitId, QubitId, HamiltonianTerm], float] = field(
        default_factory=dict
    )
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId, SetControl], list] = field(
        default_factory=dict
    )
    """Fitted parameters for Hamiltonian Terms values for different amplitudes."""


@dataclass(kw_only=True)
class HamiltonianTomographyCRAmplData(HamiltonianTomographyData):
    """Data structure for CR Amplitude."""

    cr_duration: float
    control_phase: float
    target_amplitude: float | None
    target_phase: float


def _acquisition(
    params: HamiltonianTomographyCRAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We store the probability of the control qubit and the expectation value of the target qubit.
    """

    data = HamiltonianTomographyCRAmplData(
        echo=params.echo,
        cr_duration=params.pulse_duration,
        control_phase=params.control_phase,
        target_amplitude=params.target_amplitude,
        target_phase=params.target_phase,
    )

    # update the CR channel with the target qubit frequency.
    updates = [
        {
            platform.qubits[c].drive_extra[t]: {
                "frequency": platform.config(platform.qubits[t].drive).frequency
            }
            for c, t in targets
        }
    ]

    # We create one sequence for each combination of basis and control state,
    # and we sweep the amplitude of the CR pulse for each of them; unrolling is not performed
    for basis in Basis:
        for setup in SetControl:
            sequence, cr_pulses, _, _, _ = cross_resonance_experiment(
                platform=platform,
                pair_list=targets,
                duration=params.pulse_duration,
                ctrl_ampl=0.0,  # this is the swept param
                ctrl_phase=params.control_phase,
                targ_ampl=params.target_amplitude,
                targ_phase=params.target_phase,
                basis=basis,
                setup=setup,
                echo=params.echo,
            )

            ampl_parsweepers = ParallelSweepers(
                [
                    Sweeper(
                        parameter=Parameter.amplitude,
                        range=params.amplitude_range,
                        pulses=[cr_pulses[pair][0] for pair in targets],
                    )
                ]
            )
            if params.echo:
                # sweeping over the out-of-phase signal (refocusing)
                echo_ampl_range = tuple(-x for x in params.amplitude_range)
                ampl_parsweepers += ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.amplitude,
                            range=echo_ampl_range,
                            pulses=[cr_pulses[pair][1] for pair in targets],
                        )
                    ]
                )

            results = platform.execute(
                [sequence],
                [ampl_parsweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            for ctrl, trg in targets:
                target_acq_handle = list(
                    sequence.channel(platform.qubits[trg].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[ctrl].acquisition)
                )[-1].id

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyType,
                    (ctrl, trg, basis, setup),
                    dict(
                        x=np.arange(*params.amplitude_range),
                        prob_target=1 - 2 * prob_target,
                        error_target=(
                            2 * np.sqrt(prob_target * (1 - prob_target) / params.nshots)
                        ).tolist(),
                        prob_control=1 - 2 * prob_control,
                        error_control=(
                            2
                            * np.sqrt(prob_control * (1 - prob_control) / params.nshots)
                        ).tolist(),
                    ),
                )
    return data


def _fit(
    data: HamiltonianTomographyCRAmplData,
) -> HamiltonianTomographyCRAmplResults:
    """Post-processing function for HamiltonianTomographyCRAmpl.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    fitted_parameters, cr_gate_ampls = tomography_cr_fit(data=data)
    hamiltonian_terms = {}
    for pair in data.pairs:
        hamiltonian_terms |= extract_hamiltonian_terms(
            pair=pair, fitted_parameters=fitted_parameters
        )

    return HamiltonianTomographyCRAmplResults(
        echo=data.echo,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        cr_amplitudes=cr_gate_ampls,
        cr_duration=data.cr_duration,
        control_phase=data.control_phase,
        target_amplitude=data.target_amplitude,
        target_phase=data.target_phase,
    )


def _plot(
    data: HamiltonianTomographyCRAmplData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplResults | None = None,
):
    """Plotting function for HamiltonianTomographyCRAmpl."""
    figs, fitting_report = tomography_cr_plot(data, target, fit)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                11 * [target],
                (
                    [f"{term.name} [MHz]" for term in HamiltonianTerm]
                    + [
                        "CR duration (ns)",
                        "Control amplitude (a.u.)",
                        "Control phase (rad)",
                        "Target amplitude (a.u.)",
                        "Target phase (rad)",
                    ]
                ),
                (
                    [
                        fit.hamiltonian_terms[target[0], target[1], term] * kilo
                        for term in HamiltonianTerm
                    ]
                    + [fit.cr_duration]
                    + [
                        fit.cr_amplitudes[target]
                        if target in fit.cr_amplitudes
                        else None
                    ]
                    + [fit.control_phase, fit.target_amplitude, fit.target_phase]
                ),
            )
        )
    else:
        fitting_report = ""
    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCRAmplResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    # check if the resulting fit was succsessfull
    if target in results.cr_amplitudes:
        update_cnot_from_fit(
            platform=platform,
            pair=target,
            cr_duration=results.cr_duration,
            cr_ampl=results.cr_amplitudes[target],
            control_phase=results.control_phase,
            canc_ampl=results.target_amplitude,
            canc_phase=results.target_phase,
            echo_flag=results.echo,
        )


cr_amplitude = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
We repeat the measurement twice for each target qubit, once with the control qubit in state 0
and once with the control qubit in state 1.
"""
