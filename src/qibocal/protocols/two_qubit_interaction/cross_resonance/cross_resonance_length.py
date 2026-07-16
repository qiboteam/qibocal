"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit after the application of a cross resonance sequence.
The CR pulses are played on the control drive channel with frequency set to the frequency of the target drive channel.
"""

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
    Protocol,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html

from .cr_parent_classes import (
    Basis,
    HamiltonianTerm,
    HamiltonianTomographyData,
    HamiltonianTomographyParameters,
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
    QubitId,
    cross_resonance_experiment,
    ro_delay_range,
    update_cnot_from_fit,
)

__all__ = ["cr_length"]


@dataclass(kw_only=True)
class HamiltonianTomographyCRLengthParameters(HamiltonianTomographyParameters):
    """HamiltonianTomographyCRLength runcard inputs."""

    pulse_amplitude: float
    """CR pulse amplitude"""
    phase: float = 0.0
    """Phase of CR pulse."""
    target_amplitude: float | None = None
    """Amplitude of cancellation pulse."""
    target_phase: float = 0
    """Phase of target pulse."""


@dataclass(kw_only=True)
class HamiltonianTomographyCRLengthResults(HamiltonianTomographyResults):
    """HamiltonianTomographyCRLength outputs."""

    control_amplitude: float
    control_phase: float = 0
    target_amplitude: float | None = None
    target_phase: float = 0
    hamiltonian_terms: dict[tuple[QubitId, QubitId, HamiltonianTerm], float] = field(
        default_factory=dict
    )
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId, SetControl], list] = field(
        default_factory=dict
    )
    """Fitted parameters from X,Y,Z expectation values."""


@dataclass(kw_only=True)
class HamiltonianTomographyCRLengthData(HamiltonianTomographyData):
    """Data structure for CR length."""

    control_amplitude: float
    control_phase: float
    target_amplitude: float | None
    target_phase: float


def _acquisition(
    params: HamiltonianTomographyCRLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRLengthData:
    """Data acquisition for Hamiltonian tomography CR protocol."""

    data = HamiltonianTomographyCRLengthData(
        echo=params.echo,
        control_amplitude=params.pulse_amplitude,
        control_phase=params.phase,
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
    # and we sweep the duration of the CR pulse for each of them; unrolling is not performed
    for basis in Basis:
        for setup in SetControl:
            sequence, cr_pulses, cr_target_pulses, cr_delays, ro_delays = (
                cross_resonance_experiment(
                    platform=platform,
                    pair_list=targets,
                    duration=0.0,  # this is the swept param
                    ctrl_ampl=params.pulse_amplitude,
                    ctrl_phase=params.phase,
                    targ_ampl=params.target_amplitude,
                    targ_phase=params.target_phase,
                    basis=basis,
                    setup=setup,
                    echo=params.echo,
                    interpolated_sweeper=params.interpolated_sweeper,
                )
            )

            if params.interpolated_sweeper:
                duration_parsweepers = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration_interpolated,
                            range=params.duration_range,
                            pulses=cr_pulses[pair] + cr_target_pulses[pair],
                        )
                        for pair in targets
                    ]
                )

            else:
                duration_parsweepers = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration,
                            range=params.duration_range,
                            pulses=cr_pulses[pair]
                            + cr_target_pulses[pair]
                            + cr_delays[pair],
                        )
                        for pair in targets
                    ]
                )
                ro_parsweepers = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration,
                            range=ro_delay_range(
                                cr_pulse_duration_range=params.duration_range,
                                echo=params.echo,
                                cntl_setup=setup,
                                control=c,
                                platform=platform,
                            ),
                            pulses=ro_delays[(c, t)],
                        )
                        for c, t in targets
                    ]
                )
                duration_parsweepers += ro_parsweepers

            results = platform.execute(
                [sequence],
                [duration_parsweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )

            for control, target in targets:
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyType,
                    (control, target, basis, setup),
                    dict(
                        x=np.arange(*params.duration_range),
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
    data: HamiltonianTomographyCRLengthData,
) -> HamiltonianTomographyCRLengthResults:
    """Post-processing function for HamiltonianTomographyCRLength.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    fitted_parameters, cr_gate_lengths = tomography_cr_fit(
        data=data,
    )
    hamiltonian_terms = {}
    for pair in data.pairs:
        hamiltonian_terms |= extract_hamiltonian_terms(
            pair=pair, fitted_parameters=fitted_parameters
        )

    return HamiltonianTomographyCRLengthResults(
        echo=data.echo,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        cr_lengths=cr_gate_lengths,
        control_amplitude=data.control_amplitude,
        control_phase=data.control_phase,
        target_amplitude=data.target_amplitude,
        target_phase=data.target_phase,
    )


def _plot(
    data: HamiltonianTomographyCRLengthData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRLengthResults | None = None,
):
    """Plotting function for HamiltonianTomographyCRLength."""
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
                    + [fit.cr_lengths[target] if target in fit.cr_lengths else None]
                    + [
                        fit.control_amplitude,
                        fit.control_phase,
                        fit.target_amplitude,
                        fit.target_phase,
                    ]
                ),
            )
        )
    else:
        fitting_report = ""
    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCRLengthResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):

    # check if the resulting fit was succsessfull
    if target in results.cr_lengths:
        update_cnot_from_fit(
            platform=platform,
            pair=target,
            cr_duration=results.cr_lengths[target],
            cr_ampl=results.control_amplitude,
            control_phase=results.control_phase,
            canc_ampl=results.target_amplitude,
            canc_phase=results.target_phase,
            echo_flag=results.echo,
        )


cr_length = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""HamiltonianTomographyCRLength Protocol object.

We measure the expectation values X,Y and Z on the target qubit after
applying the CR sequence specified by the input parameters. We repeat the
measurement twice for each target qubit, once with the control qubit in state 0
and once with the control qubit in state 1.

Check http://login.qrccluster.com:9000/u7Xw0C4_Rti6s2JdJ_HI4g== for a good experiment result found on emulator with 1% of classical crosstalk between
the two qubits.
"""
