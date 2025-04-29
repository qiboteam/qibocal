"""Protocol for cross resonance with sweep on amplitude of the pulse."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from ....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from ....calibration import CalibrationPlatform
from ....result import probability
from ...rabi.utils import fit_amplitude_function, rabi_amplitude_function
from .utils import SetControl, cr_fit, cr_plot, cr_sequence

CrossResonanceAmplitudeType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("x", np.float64),
    ]
)
"""Custom dtype for cross resonance amplitude."""


@dataclass
class CrossResonanceAmplitudeParameters(Parameters):
    """CrossResonanceAmplitude runcard inputs."""

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
class CrossResonanceAmplitudeResults(Results):
    """CrossResonanceAmplitude outputs."""

    fitted_parameters: dict[tuple[QubitPairId, str], list] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class CrossResonanceAmplitudeData(Data):
    """Data structure for CR amplitude."""

    data: dict[
        tuple[QubitId, QubitId, str], npt.NDArray[CrossResonanceAmplitudeType]
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrossResonanceAmplitudeData:
    """Data acquisition for CR amplitude.

    We measure the probabilities of both the target and the control qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.
    """

    data = CrossResonanceAmplitudeData()

    for pair in targets:
        control, target = pair
        pair = (control, target)
        for setup in SetControl:
            sequence, cr_pulses, _ = cr_sequence(
                platform=platform,
                control=control,
                target=target,
                setup=setup,
                amplitude=params.min_amp,
                duration=params.pulse_duration,
                echo=params.echo,
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
                CrossResonanceAmplitudeType,
                (control, target, setup),
                dict(
                    x=sweeper.values,
                    prob_target=prob_target,
                    error_target=np.sqrt(
                        prob_target * (1 - prob_target) / params.nshots
                    ).tolist(),
                    prob_control=prob_control,
                    error_control=np.sqrt(
                        prob_control * (1 - prob_control) / params.nshots
                    ).tolist(),
                ),
            )
    return data


def _fit(
    data: CrossResonanceAmplitudeData,
) -> CrossResonanceAmplitudeResults:
    """Post-processing function for CrossResonanceAmplitude.

    The target qubit probabilities are fitted with cosine oscillations.

    """

    fitted_parameters = cr_fit(data=data, fitting_function=fit_amplitude_function)
    return CrossResonanceAmplitudeResults(
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CrossResonanceAmplitudeData,
    target: QubitPairId,
    fit: CrossResonanceAmplitudeResults,
):
    """Plotting function for CrossResonanceAmplitude."""
    figs, fitting_report = cr_plot(
        data=data, target=target, fit=fit, fitting_function=rabi_amplitude_function
    )
    figs[0].update_layout(
        xaxis_title="Cross resonance pulse amplitude [a.u.]",
        yaxis_title="Excited state population",
    )
    return figs, fitting_report


cross_resonance_amplitude = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
