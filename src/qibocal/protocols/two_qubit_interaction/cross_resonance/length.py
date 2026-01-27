"""Protocol to measure CR interaction varying drive amplitude."""

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
from ...rabi.utils import fit_length_function, rabi_length_function
from ...utils import table_dict, table_html
from .utils import SetControl, cr_fit, cr_plot, cr_sequence

CrossResonanceLengthType = np.dtype(
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
class CrossResonanceLengthParameters(Parameters):
    """CrossResonanceLength runcard inputs."""

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
class CrossResonanceLengthResults(Results):
    """CrossResonanceLength outputs."""

    effective_coupling: dict[tuple[QubitId, QubitId], float] = field(
        default_factory=dict
    )
    fitted_parameters: dict[tuple[QubitPairId, str], list] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class CrossResonanceLengthData(Data):
    """Data structure for CR length."""

    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    data: dict[tuple[QubitId, QubitId, str], npt.NDArray[CrossResonanceLengthType]] = (
        field(default_factory=dict)
    )
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}


def _acquisition(
    params: CrossResonanceLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrossResonanceLengthData:
    """Data acquisition for cross resonance protocol."""

    data = CrossResonanceLengthData()

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
        for setup in SetControl:
            sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                platform=platform,
                control=control,
                target=target,
                setup=setup,
                amplitude=params.pulse_amplitude,
                duration=params.pulse_duration_end,
                interpolated_sweeper=params.interpolated_sweeper,
                echo=params.echo,
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

            try:
                cr_frequency = {
                    platform.qubits[control].drive_extra[target]: {
                        "frequency": platform.config(
                            platform.qubits[target].drive
                        ).frequency
                    }
                }
                target_offset_sweeper = control_offset_sweeper = None
            except Exception:
                cr_frequency = {
                    platform.qubits[control].drive_extra[(1, 2)]: {
                        "frequency": platform.config(
                            platform.qubits[target].drive
                        ).frequency
                    }
                }
                target_channel = platform.qubits[target].flux
                target_offset = platform.config(target_channel).offset
                target_offset_sweeper = Sweeper(
                    parameter=Parameter.offset,
                    values=np.array([target_offset]),
                    channels=[target_channel],
                )
                control_channel = platform.qubits[control].flux
                control_offset = platform.config(control_channel).offset
                control_offset_sweeper = Sweeper(
                    parameter=Parameter.offset,
                    values=np.array([control_offset]),
                    channels=[control_channel],
                )

            updates = []
            updates.append(cr_frequency)

            if target_offset_sweeper is not None:
                # execute the sweep
                results = platform.execute(
                    [sequence],
                    [[target_offset_sweeper], [control_offset_sweeper], [sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.SINGLESHOT,
                    updates=updates,
                )
            else:
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

            if target_offset_sweeper is not None:
                prob_target = prob_target[0][0]
                prob_control = prob_control[0][0]

            data.register_qubit(
                CrossResonanceLengthType,
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
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceLengthData,
) -> CrossResonanceLengthResults:
    """Post-processing function for CrossResonanceLength.

    After fitting the data with dumped cosine function, the effective coupling
    is computed as specified in https://arxiv.org/pdf/1905.11480.

    """
    fitted_parameters = cr_fit(data=data, fitting_function=fit_length_function)
    effective_coupling = {}
    for pair in data.pairs:
        try:
            effective_coupling[pair] = (
                1 / fitted_parameters[pair[0], pair[1], SetControl.X][2]
                - 1 / fitted_parameters[pair[0], pair[1], SetControl.Id][2]
            ) / 2
        except KeyError:  # pragma: no cover
            pass
    return CrossResonanceLengthResults(
        effective_coupling=effective_coupling,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CrossResonanceLengthData,
    target: QubitPairId,
    fit: CrossResonanceLengthResults,
):
    """Plotting function for CrossResonanceLength."""
    figs, fitting_report = cr_plot(
        data=data, target=target, fit=fit, fitting_function=rabi_length_function
    )
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                [target],
                [
                    "Effective coupling [MHz]",
                ],
                [fit.effective_coupling[target] * kilo],
            )
        )

    figs[0].update_layout(
        xaxis_title="Cross resonance pulse duration [ns]",
        yaxis_title="Excited state population",
    )
    return figs, fitting_report


cross_resonance_length = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
