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
from .....result import probability
from ....utils import table_dict, table_html
from ..utils import Basis, SetControl, cr_sequence
from .utils import (
    EPS,
    HamiltonianTerm,
    extract_hamiltonian_terms,
    tomography_cr_fit,
    tomography_cr_plot,
)

HamiltonianTomographyCRAmplitudeType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("x", np.float64),
    ]
)
"""Custom dtype for CR length."""


@dataclass
class HamiltonianTomographyCRAmplitudeParameters(Parameters):
    """HamiltonianTomographyCRLength runcard inputs."""

    pulse_duration: int
    """Duration of CR pulse [ns]."""
    pulse_amplitude_start: float
    """Initial amplitude of CR pulse [a.u.]."""
    pulse_amplitude_end: float
    """Final amplitude of CR pulse [a.u.]."""
    pulse_amplitude_step: float
    """Step CR pulse duration [a.u.]."""
    phase: float = 0.0
    """Phase of CR pulse."""
    target_amplitude: float = 0
    """Amplitude of cancellation pulse [a.u.]."""
    target_phase: float = 0
    """Phase of target pulse."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """

    @property
    def amplitude_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_amplitude_start,
            self.pulse_amplitude_end,
            self.pulse_amplitude_step,
        )


@dataclass
class HamiltonianTomographyCRAmplitudeResults(Results):
    """HamiltonianTomographyCRLength outputs."""

    echo: bool
    hamiltonian_terms: dict[QubitId, QubitId, HamiltonianTerm] = field(
        default_factory=dict
    )
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId, Basis, SetControl], list] = field(
        default_factory=dict
    )
    """Fitted parameters from X,Y,Z expectation values."""
    cr_amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Estimated_duration of CR gate."""
    cr_duration: int = 0
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    native: Literal["CNOT"] = "CNOT"
    """Two qubit interaction to be calibrated."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRAmplitudeData(Data):
    """Data structure for CR length."""

    echo: bool | None = None
    cr_duration: int = 0
    control_phase: float = 0
    target_amplitude: float = 0
    target_phase: float = 0
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRAmplitudeType],
    ] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}


def _acquisition(
    params: HamiltonianTomographyCRAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplitudeData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRAmplitudeData()
    data.echo = params.echo
    data.cr_duration = params.pulse_duration
    data.control_phase = params.phase
    data.target_amplitude = params.target_amplitude
    data.target_phase = params.target_phase

    for pair in targets:
        control, target = pair
        pair = (control, target)
        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, _, _ = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    amplitude=params.pulse_amplitude_end,
                    phase=params.phase,
                    target_amplitude=params.target_amplitude,
                    target_phase=params.target_phase,
                    duration=params.pulse_duration,
                    echo=params.echo,
                    setup=setup,
                    basis=basis,
                )

                sweeper = Sweeper(
                    parameter=Parameter.amplitude,
                    values=params.amplitude_range,
                    pulses=cr_pulses,
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
                    averaging_mode=AveragingMode.SINGLESHOT,
                    updates=updates,
                )
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id

                prob_target = probability(results[target_acq_handle], state=1).ravel()
                prob_control = probability(results[control_acq_handle], state=1).ravel()

                # TODO: possibly drop control probablity even if it might be useful later on
                # to compute leakage
                data.register_qubit(
                    HamiltonianTomographyCRAmplitudeType,
                    (control, target, basis, setup),
                    dict(
                        x=sweeper.values,
                        prob_target=1 - 2 * prob_target,
                        error_target=(
                            2
                            * np.sqrt(
                                EPS + prob_target * (1 - prob_target) / params.nshots
                            )
                        ).tolist(),
                        prob_control=prob_control,
                        error_control=np.sqrt(
                            EPS + prob_control * (1 - prob_control) / params.nshots
                        ).tolist(),
                    ),
                )

    return data


def _fit(
    data: HamiltonianTomographyCRAmplitudeData,
) -> HamiltonianTomographyCRAmplitudeResults:
    """Post-processing function for HamiltonianTomographyCRLength.

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

    return HamiltonianTomographyCRAmplitudeResults(
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
    data: HamiltonianTomographyCRAmplitudeData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplitudeResults,
):
    """Plotting function for HamiltonianTomographyCRLength."""
    figs, fitting_report = tomography_cr_plot(data, target, fit)
    figs[0].update_layout(
        xaxis3_title="CR pulse amplitude [a.u.]",
    )
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                6 * [target],
                [f"{term.name} [MHz]" for term in HamiltonianTerm],
                [
                    fit.hamiltonian_terms[target[0], target[1], term] * kilo
                    for term in HamiltonianTerm
                ],
            )
        )
    else:
        fitting_report = ""
    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCRAmplitudeResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    target = target[::-1] if target not in results.cr_amplitudes else target

    new_cr_seq, _, _, _ = cr_sequence(
        platform=platform,
        control=target[0],
        target=target[1],
        amplitude=results.cr_amplitudes[target],
        duration=results.cr_duration,
        phase=results.control_phase,
        target_amplitude=results.target_amplitude,
        target_phase=results.target_phase,
        echo=results.echo,
        basis=Basis.Y,
    )

    new_cr_seq.insert(
        -4, (platform.qubits[target[0]].drive, VirtualZ(phase=-np.pi / 2))
    )

    getattr(update, f"{results.native}_sequence")(new_cr_seq, platform, target)


hamiltonian_tomography_cr_amplitude = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""HamiltonianTomographyCRAmplitude Routine object."""
