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
from .utils import (
    HamiltonianTerm,
    extract_hamiltonian_terms,
    tomography_cr_fit,
    tomography_cr_plot,
)

__all__ = ["hamiltonian_tomography_cr_length"]

HamiltonianTomographyCRLengthType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("x", np.int64),
    ]
)
"""Custom dtype for CR length."""


@dataclass
class HamiltonianTomographyCRLengthParameters(Parameters):
    """HamiltonianTomographyCRLength runcard inputs."""

    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""
    pulse_amplitude: float
    """CR pulse amplitude"""
    phase: float = 0.0
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

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRLengthResults(Results):
    """HamiltonianTomographyCRLength outputs."""

    echo: bool
    hamiltonian_terms: dict[tuple[QubitId, QubitId, HamiltonianTerm], float] = field(
        default_factory=dict
    )
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId, Basis, SetControl], list] = field(
        default_factory=dict
    )
    """Fitted parameters from X,Y,Z expectation values."""
    cr_lengths: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Estimated durations of CR gate."""
    control_amplitude: float = 0
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    native: Literal["CNOT"] = "CNOT"
    """Two qubit interaction to be calibrated."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRLengthData(Data):
    """Data structure for CR length."""

    echo: bool
    control_amplitude: float = 0
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRLengthType],
    ] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}


def _acquisition(
    params: HamiltonianTomographyCRLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRLengthData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRLengthData(echo=params.echo)
    data.control_amplitude = params.pulse_amplitude
    data.control_phase = params.phase
    data.target_amplitude = params.target_amplitude
    data.target_phase = params.target_phase

    for pair in targets:
        control, target = pair
        pair = (control, target)
        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    amplitude=params.pulse_amplitude,
                    phase=params.phase,
                    target_amplitude=params.target_amplitude,
                    target_phase=params.target_phase,
                    duration=params.pulse_duration_end,
                    interpolated_sweeper=params.interpolated_sweeper,
                    echo=params.echo,
                    setup=setup,
                    basis=basis,
                )

                if params.interpolated_sweeper:
                    sweeper = Sweeper(
                        parameter=Parameter.duration_interpolated,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses,
                    )
                else:
                    sweeper = Sweeper(
                        parameter=Parameter.duration,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses + delays,
                    )

                updates = []
                updates.append(
                    {
                        platform.qubits[control].drive_extra[(1, 2)]: {
                            "frequency": platform.config(
                                platform.qubits[target].drive
                            ).frequency
                        }
                    }
                )

                results = platform.execute(
                    [sequence],
                    [[sweeper]],
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
                    HamiltonianTomographyCRLengthType,
                    (control, target, basis, setup),
                    dict(
                        x=sweeper.values,
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
    fit: HamiltonianTomographyCRLengthResults,
):
    """Plotting function for HamiltonianTomographyCRLength."""
    figs, fitting_report = tomography_cr_plot(data, target, fit)
    figs[0].update_layout(
        xaxis3_title="CR pulse length [ns]",
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
    target = target[::-1] if target not in results.cr_lengths else target

    new_cr_seq, _, _, _ = cr_sequence(
        platform=platform,
        control=target[0],
        target=target[1],
        amplitude=results.control_amplitude,
        duration=results.cr_lengths[target],
        phase=results.control_phase,
        target_amplitude=results.target_amplitude,
        target_phase=results.target_phase,
        echo=results.echo,
        setup=SetControl.Id,
        basis=Basis.Z,
    )
    new_cr_seq = new_cr_seq[:-2]  # remove acquisition pulses

    new_cr_seq.insert(
        -2,
        (
            platform.qubits[target[1]].drive,
            platform.natives.single_qubit[target[1]].R(theta=3 * np.pi / 2, phi=0)[0][
                1
            ],
        ),
    )
    new_cr_seq.insert(
        -2, (platform.qubits[target[0]].drive, VirtualZ(phase=-np.pi / 2))
    )

    getattr(update, f"{results.native.lower()}_sequence")(new_cr_seq, platform, target)


hamiltonian_tomography_cr_length = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""HamiltonianTomographyCRLength Routine object."""

"""
Check http://login.qrccluster.com:9000/u7Xw0C4_Rti6s2JdJ_HI4g== for a good experiment result found on emulator with 1% of classical crosstalk between
the two qubits.
"""
