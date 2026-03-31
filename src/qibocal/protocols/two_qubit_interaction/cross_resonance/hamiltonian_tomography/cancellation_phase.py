"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
    VirtualZ,
)
from scipy.constants import kilo

from ..... import update
from .....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from .....calibration import CalibrationPlatform
from ....utils import table_dict, table_html
from ..utils import Basis, SetControl, cr_sequence
from .length import HamiltonianTomographyCRLengthData
from .utils import (
    HamiltonianTerm,
    cancellation_calibration_plot,
    cancellation_phase_fit,
    reconstruct_full_hamiltonian_terms,
)

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


@dataclass
class HamiltonianTomographyCANCPhaseParameters(Parameters):
    """HamiltonianTomographyCANCPhase runcard inputs."""

    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""
    control_phase_step: float
    """Step CR pulse phase."""
    control_phase: float = 0
    """Initial phase of CR pulse."""
    control_phase_end: float = 2 * np.pi
    """Final phase of CR pulse."""
    control_amplitude: float | None = None
    """Initial amplitude of CR pulse."""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def phase_range(self) -> np.ndarray:
        """Amplitude range for CR pulses."""
        return np.arange(
            self.control_phase,
            self.control_phase_end,
            self.control_phase_step,
        )

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCANCPhaseResults(Results):
    """HamiltonianTomographyCANCPhase outputs."""

    echo: bool

    control_amplitude: float

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

    native: Literal["CNOT"] = "CNOT"
    """Two qubit interaction to be calibrated."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))

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
class HamiltonianTomographyCANCPhaseData(Data):
    """Data structure for CR Amplitude."""

    echo: bool
    control_amplitude: float | None = None
    phases: list | None = None
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCANCPhaseType],
    ] = field(default_factory=dict)
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}

    def select_phase(self, phase: float) -> HamiltonianTomographyCRLengthData:
        new_data = HamiltonianTomographyCRLengthData(
            echo=self.echo,
        )
        new_data.data = {k: d[d.phase == phase] for k, d in self.data.items()}
        return new_data

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit."""
        duration_list = data_dict["x"]
        phase_list = data_dict["phase"]
        size = len(duration_list) * len(phase_list)
        ar = np.empty(size, dtype=dtype)
        durations, phases = np.meshgrid(duration_list, phase_list)
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
        phases=params.phase_range.astype(float).tolist(),
        control_amplitude=params.control_amplitude,
        verbose_plot=params.verbose_plot,
    )

    for pair in targets:
        control, target = pair

        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    amplitude=params.control_amplitude,
                    phase=params.control_phase_end,
                    target_amplitude=0,
                    target_phase=0,
                    duration=params.pulse_duration_end,
                    echo=params.echo,
                    setup=setup,
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
                phase_sweepers = [
                    Sweeper(
                        parameter=Parameter.relative_phase,
                        values=params.phase_range,
                        pulses=[cr_pulses[0]],
                    )
                ]
                if params.echo:
                    phase_sweepers.append(
                        Sweeper(
                            parameter=Parameter.relative_phase,
                            values=params.phase_range + np.pi,
                            pulses=[cr_pulses[1]],
                        )
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
                    [phase_sweepers, [length_sweeper]],
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

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyCANCPhaseType,
                    (control, target, basis, setup),
                    dict(
                        x=length_sweeper.values,
                        phase=phase_sweepers[0].values,
                        prob_target=1 - 2 * prob_target,
                        error_target=2
                        * np.sqrt(prob_target * (1 - prob_target) / params.nshots),
                        prob_control=prob_control,
                        error_control=np.sqrt(
                            prob_control * (1 - prob_control) / params.nshots
                        ),
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
        control_amplitude=data.control_amplitude,
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
    fit: HamiltonianTomographyCANCPhaseResults,
):
    """Plotting function for HamiltonianTomographyCANCPhase."""
    figs, fitting_report = cancellation_calibration_plot(data, target, fit)

    if fit.verbose_plot:
        from qibocal.protocols.two_qubit_interaction.cross_resonance.hamiltonian_tomography.length import (
            HamiltonianTomographyCRLengthResults,
        )
        from qibocal.protocols.two_qubit_interaction.cross_resonance.hamiltonian_tomography.utils import (
            tomography_cr_plot,
        )

        for phi in data.phases:
            selected_ham_terms = fit.select_pair_and_phase_ham_params(phi, target)
            ampl_data = data.select_phase(phi)
            ham_tom_fit = HamiltonianTomographyCRLengthResults(
                echo=fit.echo,
                hamiltonian_terms=selected_ham_terms,
                fitted_parameters=fit.hamiltonian_tom_params[phi],
                cr_lengths=fit.cr_lengths[phi],
            )
            f, _ = tomography_cr_plot(ampl_data, target, ham_tom_fit)
            figs += f
            if ham_tom_fit is not None:
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

    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCANCPhaseResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    # here is updated the full CNOT Pulse Sequence, which is composed by a CR sequence followe by a X_pi/2 and Z_(-pi/2) rotations on
    # target and control qubit respectively

    target = target[::-1] if target not in results.cancellation_pulse_phases else target

    new_cr_seq, _, _, _ = cr_sequence(
        platform=platform,
        control=target[0],
        target=target[1],
        amplitude=results.control_amplitude,
        duration=None,
        phase=results.cancellation_pulse_phases[target]["control"],
        target_amplitude=0,
        target_phase=results.cancellation_pulse_phases[target]["target"],
        echo=results.echo,
        setup=SetControl.Id,
        basis=Basis.Y,
    )
    new_cr_seq = new_cr_seq[:-2]  # remove acquisition pulses

    new_cr_seq.insert(
        -2, (platform.qubits[target[0]].drive, VirtualZ(phase=-np.pi / 2))
    )

    getattr(update, f"{results.native}_sequence")(new_cr_seq, platform, target)


hamiltonian_tomography_canc_phase = Routine(_acquisition, _fit, _plot, _update)
"""HamiltonianTomographyCANCPhase Routine object."""

"""See http://login.qrccluster.com:9000/bf6RezP1SpCnI861v6UNpA== for an example run on the emulator.
"""
