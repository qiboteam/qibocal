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
from .....result import probability
from ....rabi.utils import fit_amplitude_function, rabi_amplitude_function
from ..utils import Basis, SetControl, cr_sequence
from .utils import tomography_cr_fit, tomography_cr_plot

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
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    pulse_duration: int
    """CR pulse duration in ns."""
    echo: bool = False
    """Apply echo sequence or not."""

    @property
    def amplitude_range(self):
        return np.arange(self.min_amp, self.max_amp, self.step_amp)


@dataclass
class HamiltonianTomographyCRAmplitudeResults(Results):
    """HamiltonianTomographyCRAmplitude outputs."""

    fitted_parameters: dict[tuple[QubitId, QubitId, Basis, SetControl], list] = field(
        default_factory=dict
    )

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRAmplitudeData(Data):
    """Data structure for CR length."""

    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    data: dict[
        tuple[QubitId, QubitId, str], npt.NDArray[HamiltonianTomographyCRAmplitudeType]
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: HamiltonianTomographyCRAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplitudeData:
    """Data acquisition for cross resonance protocol."""

    data = HamiltonianTomographyCRAmplitudeData()

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
                    amplitude=params.min_amp,
                    duration=params.pulse_duration,
                    echo=params.echo,
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
                    HamiltonianTomographyCRAmplitudeType,
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
    data: HamiltonianTomographyCRAmplitudeData,
) -> HamiltonianTomographyCRAmplitudeResults:
    """Post-processing function for HamiltonianTomographyCRAmplitude."""
    fitted_parameters = tomography_cr_fit(
        data=data, fitting_function=fit_amplitude_function
    )
    return HamiltonianTomographyCRAmplitudeResults(fitted_parameters)


def _plot(
    data: HamiltonianTomographyCRAmplitudeData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplitudeResults,
):
    """Plotting function for HamiltonianTomographyCRAmplitude."""
    figs, fitting_report = tomography_cr_plot(
        data=data,
        target=target,
        fit=fit,
        fitting_function=rabi_amplitude_function,
    )
    figs[0].update_layout(
        xaxis3_title="CR pulse amplitude [a.u.]",
    )
    return figs, fitting_report


hamiltonian_tomography_cr_amplitude = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
