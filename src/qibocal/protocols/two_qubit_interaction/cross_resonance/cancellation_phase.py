"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
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
    QubitId,
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
    SetControl,
)
from .cross_resonance_processing import (
    cancellation_phase_fit,
    reconstruct_full_hamiltonian_terms,
)
from .plotting import cancellation_calibration_plot
from .utils import (
    cross_resonance_experiment,
    cross_resonance_pulses,
    ro_delay_range,
    update_cnot_from_fit,
)

__all__ = ["cancellation_phase_tuning"]

HamiltonianTomographyCANCPhaseType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("phase", np.float64),
        ("x", np.int64),
    ]
)
"""Custom dtype for Cancellation phase."""


@dataclass(kw_only=True)
class HamiltonianTomographyCANCPhaseParameters(HamiltonianTomographyParameters):
    """HamiltonianTomographyCANCPhase runcard inputs."""

    phase_range: tuple[float, float, float]
    """CR pulse phase range for control qubit."""
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""


@dataclass(kw_only=True)
class HamiltonianTomographyCANCPhaseResults(HamiltonianTomographyResults):
    """HamiltonianTomographyCANCPhase outputs."""

    hamiltonian_terms: dict[
        QubitPairId, list[tuple[float, dict[HamiltonianTerm, float]]]
    ] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters from Hamiltonian Terms values for different phases."""

    cancellation_pulse_phases: dict[QubitPairId, dict[str, float]] = field(
        default_factory=dict
    )
    """Fitted parameters for cancellation pulse phases."""
    hamiltonian_tom_params: dict[
        float, dict[tuple[QubitId, QubitId, SetControl], list[float]]
    ] = field(default_factory=dict)
    """Fitted parameters for of Hamiltonian Tomography experiment for each qubit and per amplitude value.
    Used for plotting with the `verbose_plot` option"""

    cr_lengths: dict[float, dict[tuple[QubitId, QubitId, SetControl], list[float]]] = (
        field(default_factory=dict)
    )
    """Cross resonance pulse duration found in Hamiltonian Tomography experiment for each qubit and per amplitude value.
    Used for plotting with the `verbose_plot` option"""

    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    def select_pair_and_phase_ham_params(self, phase: float, pair: QubitPairId):
        """Data
        Select and refactor Hamiltonian tomography parameters for a given amplitude and qubit pair.
        """
        selected_ham_tom_params = next(
            (
                val_ham_params
                for (phi, val_ham_params) in self.hamiltonian_terms[pair]
                if phi == phase
            ),
            None,
        )

        return reconstruct_full_hamiltonian_terms(selected_ham_tom_params, pair)


@dataclass
class HamiltonianTomographyCANCPhaseData(HamiltonianTomographyData):
    """Data structure for CANC Phase."""

    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def phases(self) -> list[float]:
        first_key = next(iter(self.data.keys()))
        return np.unique(self.data[first_key].phase).tolist()

    def select_phase(self, phase: float) -> HamiltonianTomographyData:
        new_data = HamiltonianTomographyData(echo=self.echo)
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
        ar["error_target"] = data_dict["error_target"].ravel()
        ar["prob_control"] = data_dict["prob_control"].ravel()
        ar["error_control"] = data_dict["error_control"].ravel()

        self.data[data_keys] = np.rec.array(ar)


def _acquisition(
    params: HamiltonianTomographyCANCPhaseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCANCPhaseData:
    """Data acquisition for Hamiltonian tomography CR ctrl_phaseprotocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCANCPhaseData(
        echo=params.echo,
        verbose_plot=params.verbose_plot,
    )

    updates = []
    control_ampls: dict[QubitPairId, float] = {}
    target_ampls: dict[QubitPairId, float] = {}
    for pair in targets:
        control, target = pair

        updates.append(
            {
                platform.qubits[control].drive_extra[target]: {
                    "frequency": platform.config(
                        platform.qubits[target].drive
                    ).frequency
                }
            }
        )

        cr_pulse, canc_pulse = cross_resonance_pulses(platform, control, target)
        if cr_pulse is None:
            raise ValueError(
                "CR pulse not found for control {control} and target {target}. "
                "Please check the CR pulse configuration or first run previous protocols."
            )
        control_ampls |= {pair: cr_pulse.amplitude}
        target_ampls |= {pair: canc_pulse.amplitude if canc_pulse is not None else 0.0}

    for basis in Basis:
        for setup in SetControl:
            sequence, cr_pulses, cr_target_pulses, cr_delays, ro_delays = (
                cross_resonance_experiment(
                    platform=platform,
                    pair_list=targets,
                    duration=0.0,
                    ctrl_ampl=control_ampls,
                    ctrl_phase=0.0,
                    targ_ampl=target_ampls,
                    targ_phase=0.0,
                    basis=basis,
                    setup=setup,
                    echo=params.echo,
                    interpolated_sweeper=params.interpolated_sweeper,
                )
            )

            if params.interpolated_sweeper:
                duration_parallel_sweeper = ParallelSweepers(
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
                duration_parallel_sweeper = ParallelSweepers(
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
                ro_sweeper = ParallelSweepers(
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
                duration_parallel_sweeper += ro_sweeper

            phase_sweepers = ParallelSweepers(
                [
                    Sweeper(
                        parameter=Parameter.relative_phase,
                        range=params.phase_range,
                        pulses=cr_pulses[pair],
                    )
                    for pair in targets
                ]
            )

            results = platform.execute(
                [sequence],
                [
                    duration_parallel_sweeper,
                    phase_sweepers,
                ],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )

            for ctrl, targ in targets:
                target_acq_handle = list(
                    sequence.channel(platform.qubits[targ].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[ctrl].acquisition)
                )[-1].id

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyCANCPhaseType,
                    (ctrl, targ, basis, setup),
                    dict(
                        x=np.arange(*params.duration_range),
                        phase=np.arange(*params.phase_range),
                        prob_target=1 - 2 * prob_target,
                        error_target=2
                        * np.sqrt(prob_target * (1 - prob_target) / params.nshots),
                        prob_control=1 - 2 * prob_control,
                        error_control=2
                        * np.sqrt(prob_control * (1 - prob_control) / params.nshots),
                    ),
                )

    return data


def _fit(
    data: HamiltonianTomographyCANCPhaseData,
) -> HamiltonianTomographyCANCPhaseResults:
    """Post-processing function for HamiltonianTomographyCANCPhase.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    hamiltonian_terms, fitted_parameters, pulses_phases, ham_tom_params, cr_lengths = (
        cancellation_phase_fit(
            data=data,
        )
    )

    return HamiltonianTomographyCANCPhaseResults(
        echo=data.echo,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        cancellation_pulse_phases=pulses_phases,
        hamiltonian_tom_params=ham_tom_params,
        cr_lengths=cr_lengths,
        verbose_plot=data.verbose_plot,
    )


def _plot(
    data: HamiltonianTomographyCANCPhaseData,
    target: QubitPairId,
    fit: HamiltonianTomographyCANCPhaseResults | None = None,
):
    """Plotting function for HamiltonianTomographyCANCPhase."""
    figs, fitting_report = cancellation_calibration_plot(data, target, fit)

    plot_ham_tom = True if fit is None else fit.verbose_plot

    if plot_ham_tom:
        from .plotting import (
            tomography_cr_plot,
        )

        for phi in data.phases:
            ampl_data = data.select_phase(phi)
            if fit is not None:
                selected_ham_terms = fit.select_pair_and_phase_ham_params(phi, target)
                ham_tom_fit = HamiltonianTomographyResults(
                    echo=fit.echo,
                    hamiltonian_terms=selected_ham_terms,
                    fitted_parameters=fit.hamiltonian_tom_params[phi],
                    cr_lengths=fit.cr_lengths[phi],
                )

                fitting_report += "\n" + table_html(
                    table_dict(
                        8 * [target],
                        [f"{term.name} [MHz]" for term in HamiltonianTerm]
                        + ["CR duration (ns)", "Cancellation phase [rad.]"],
                        [
                            ham_tom_fit.hamiltonian_terms[target[0], target[1], term]
                            * kilo
                            for term in HamiltonianTerm
                        ]
                        + [
                            fit.cr_lengths[phi][target]
                            if target in fit.cr_lengths[phi]
                            else None
                        ]
                        + [phi],
                    )
                )
            else:
                ham_tom_fit = None
                fitting_report = ""

            f, _ = tomography_cr_plot(ampl_data, target, ham_tom_fit)
            figs += f

    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCANCPhaseResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    # here is updated the full CNOT Pulse Sequence, which is composed by a CR sequence followe by a X_pi/2 and Z_(-pi/2) rotations on
    # target and control qubit respectively

    target = target[::-1] if target not in results.cancellation_pulse_phases else target

    # now no check is needed since the acquisition was executed correctly,
    # which means we have all parameters defined.
    cr_pulse, canc_pulse = cross_resonance_pulses(platform, target[0], target[1])
    gate_duration = cr_pulse.duration
    control_amplitude = cr_pulse.amplitude
    target_amplitude = 0.0 if canc_pulse is None else canc_pulse.amplitude

    # check if the resulting fit was succsessfull
    if target in results.cancellation_pulse_phases:
        update_cnot_from_fit(
            platform=platform,
            pair=target,
            cr_duration=gate_duration,
            cr_ampl=control_amplitude,
            control_phase=results.cancellation_pulse_phases[target]["control"],
            canc_ampl=target_amplitude,
            canc_phase=results.cancellation_pulse_phases[target]["target"],
            echo_flag=results.echo,
        )


cancellation_phase_tuning = Protocol(_acquisition, _fit, _plot, _update)
"""HamiltonianTomographyCANCPhase Protocol object."""

"""See http://login.qrccluster.com:9000/bf6RezP1SpCnI861v6UNpA== for an example run on the emulator.
"""
