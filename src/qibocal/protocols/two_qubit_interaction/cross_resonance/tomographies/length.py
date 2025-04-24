from dataclasses import dataclass, field

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
from .....result import probability
from ....rabi.utils import rabi_length_function
from ....utils import table_dict, table_html
from ..utils import Basis, SetControl, cr_sequence
from .utils import (
    HamiltonianTerm,
    extract_hamiltonian_terms,
    tomography_cr_fit,
    tomography_cr_plot,
)

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
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: float
    """CR pulse amplitude"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not."""

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRLengthResults(Results):
    """HamiltonianTomographyCRLength outputs."""

    hamiltonian_terms: dict[QubitId, QubitId, HamiltonianTerm] = field(
        default_factory=dict
    )
    fitted_parameters: dict[tuple[QubitId, QubitId, Basis, SetControl], list] = field(
        default_factory=dict
    )

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRLengthData(Data):
    """Data structure for CR length."""

    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRLengthType],
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: HamiltonianTomographyCRLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRLengthData:
    """Data acquisition for cross resonance protocol."""

    data = HamiltonianTomographyCRLengthData()

    for pair in targets:
        control, target = pair
        pair = (control, target)
        data.detuning[pair] = (
            platform.config(platform.qubits[control].drive).frequency
            - platform.config(platform.qubits[target].drive).frequency
        )
        data.anharmonicity[pair] = platform.calibration.single_qubits[
            control
        ].qubit.anharmonicity
        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    setup=setup,
                    amplitude=params.pulse_amplitude,
                    duration=params.pulse_duration_end,
                    interpolated_sweeper=params.interpolated_sweeper,
                    echo=params.echo,
                    basis=basis,
                )

                if params.interpolated_sweeper:
                    sweeper = Sweeper(
                        parameter=Parameter.duration_interpolated,
                        values=params.duration_range,
                        pulses=cr_pulses,
                    )
                else:
                    sweeper = Sweeper(
                        parameter=Parameter.duration,
                        values=params.duration_range,
                        pulses=cr_pulses + delays,
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
                # execute the sweep
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
                prob_target = probability(results[target_acq_handle], state=1)
                prob_control = probability(results[control_acq_handle], state=1)
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
    # finally, save the remaining data
    return data


def _fit(
    data: HamiltonianTomographyCRLengthData,
) -> HamiltonianTomographyCRLengthResults:
    """Post-processing function for HamiltonianTomographyCRLength."""
    fitted_parameters = tomography_cr_fit(
        data=data,
    )
    hamiltonian_terms = {}
    for pair in data.pairs:
        hamiltonian_terms |= extract_hamiltonian_terms(
            pair=pair, fitted_parameters=fitted_parameters
        )

    return HamiltonianTomographyCRLengthResults(
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: HamiltonianTomographyCRLengthData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRLengthResults,
):
    """Plotting function for HamiltonianTomographyCRLength."""
    figs, fitting_report = tomography_cr_plot(data, target, fit, rabi_length_function)
    figs[0].update_layout(
        xaxis3_title="CR pulse length [ns]",
    )
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
    return figs, fitting_report


hamiltonian_tomography_cr_length = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
