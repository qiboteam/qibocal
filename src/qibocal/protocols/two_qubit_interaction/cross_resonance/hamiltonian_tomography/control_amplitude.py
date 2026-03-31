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
)
from scipy.constants import kilo

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
from .utils import (
    HamiltonianTerm,
    extract_hamiltonian_terms,
    reconstruct_full_hamiltonian_terms,
    tomography_cr_fit,
    tomography_cr_plot,
)

HamiltonianTomographyCRAmplType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("x", np.float64),
    ]
)
"""Custom dtype for Cancellation amplitude."""


@dataclass
class HamiltonianTomographyCRAmplParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    pulse_duration: float
    """Duration of CR pulse [ns]."""
    control_ampl_start: float
    """Amplitude of cancellation pulse."""
    control_ampl_end: float
    """Final amplitude of CR pulse."""
    control_ampl_step: float
    """Step CR pulse amplitude."""
    control_phase: float | None = 0.0
    """Phase of the CR pulse on the control qubit"""
    target_phase: float | None = 0.0
    """Phase of the Cancellation pulse on the target qubit"""
    target_amplitude: float | None = 0.0
    """Amplitude of the Cancellation pulse on the target qubit"""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """

    @property
    def amplitude_range(self) -> np.ndarray:
        """Amplitude range for CR pulses."""
        return np.arange(
            self.control_ampl_start,
            self.control_ampl_end,
            self.control_ampl_step,
        )


@dataclass
class HamiltonianTomographyCRAmplResults(Results):
    """HamiltonianTomographyCRAmpl outputs."""

    echo: bool
    cr_duration: float
    hamiltonian_terms: dict[
        tuple[QubitId, QubitId], list[tuple[float, dict[HamiltonianTerm, float]]]
    ] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters for Hamiltonian Terms values for different amplitudes."""
    cr_amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Estimated amplitudes of CR gate."""
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    native: Literal["CNOT"] = "CNOT"
    """Two qubit interaction to be calibrated."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))

    def select_pair_and_ampl_ham_params(self, amplitude: float, pair: QubitPairId):
        """Data
        Select and refactor Hamiltonian tomography parameters for a given amplitude and qubit pair.
        """
        selected_ham_tom_params = next(
            (
                val_ham_params
                for (amp, val_ham_params) in self.hamiltonian_terms[pair]
                if amp == amplitude
            ),
            None,
        )

        return reconstruct_full_hamiltonian_terms(selected_ham_tom_params, pair)


@dataclass
class HamiltonianTomographyCRAmplData(Data):
    """Data structure for CR Amplitude."""

    echo: bool
    cr_duration: float
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRAmplType],
    ] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}


def _acquisition(
    params: HamiltonianTomographyCRAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRAmplData(
        echo=params.echo,
        cr_duration=params.pulse_duration,
    )

    for pair in targets:
        control, target = pair

        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, _, _ = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    duration=params.pulse_duration,
                    amplitude=params.control_ampl_end,
                    phase=params.control_phase,
                    target_amplitude=params.target_amplitude,
                    target_phase=params.target_phase,
                    echo=params.echo,
                    setup=setup,
                    basis=basis,
                )

                amp_sweeper = Sweeper(
                    parameter=Parameter.amplitude,
                    values=params.amplitude_range,
                    pulses=cr_pulses,
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
                    [[amp_sweeper]],
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
                    HamiltonianTomographyCRAmplType,
                    (control, target, basis, setup),
                    dict(
                        x=amp_sweeper.values,
                        prob_target=1 - 2 * prob_target,
                        error_target=(
                            2 * np.sqrt(prob_target * (1 - prob_target) / params.nshots)
                        ).tolist(),
                        prob_control=prob_control,
                        error_control=np.sqrt(
                            prob_control * (1 - prob_control) / params.nshots
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
    fitted_parameters, cr_gate_ampls = tomography_cr_fit(
        data=data,
    )
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
    fit: HamiltonianTomographyCRAmplResults,
):
    """Plotting function for HamiltonianTomographyCRAmpl."""
    figs, fitting_report = tomography_cr_plot(data, target, fit)
    figs[0].update_layout(
        xaxis3_title="CR pulse amplitude [a.u.]",
    )

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


hamiltonian_tomography_cr_amplitude = Routine(
    _acquisition, _fit, _plot, two_qubit_gates=True
)
"""HamiltonianTomographyCRAmplitude Routine object."""
