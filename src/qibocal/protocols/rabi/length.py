from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import chi2_reduced
from .acquisition import check_correct_drive_lines_setup, sequence_length
from .parent_classes import (
    RabiData,
    RabiLengthParameters,
    RabiResults,
)
from .processing import (
    fit_length_function,
    plot_probabilities,
    rabi_initial_guess,
    rabi_length_function,
    update_rabi_parameters,
)

__all__ = ["rabi_length"]


RabiLenClassType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi duration classification."""


@dataclass
class RabiLengthClassificationData(RabiData):
    """RabiLength acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RabiLenClassType]] = field(default_factory=dict)
    """Raw data acquired for classification experiment."""


def _acquisition(
    params: RabiLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthClassificationData:
    r"""
    Data acquisition for RabiLength Classification Experiment.
    """

    drive_lines = check_correct_drive_lines_setup(
        targets=targets, input_drivelines=params.drive_lines
    )

    sequence, qd_pulses, delays, amplitudes, updates = sequence_length(
        targets=targets,
        drive_lines=drive_lines,
        platform=platform,
        pulse_ampl=params.pulse_amplitude,
        pulse_duration=None,  # in this case we are sweeping on duration
        rx90=params.rx90,
        use_align=params.interpolated_sweeper,
    )

    if params.interpolated_sweeper:
        # in this case delays is always an empty list, so it is safe to sum to qd_pulses
        sweep_param = Parameter.duration_interpolated
    else:
        sweep_param = Parameter.duration

    sweeper = Sweeper(
        parameter=sweep_param,
        range=params.duration_range,
        pulses=qd_pulses + delays,
    )

    data = RabiLengthClassificationData(
        drive_lines={t: d for t, d in zip(targets, drive_lines)},
        rx90=params.rx90,
        amplitudes=amplitudes,
    )

    # execute the sweep
    results = platform.execute(
        [sequence],
        [ParallelSweepers([sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for q in targets:
        ro_pulse = list(sequence.channel(platform.qubits[q].acquisition))[-1]
        prob = results[ro_pulse.id]
        data.register_qubit(
            RabiLenClassType,
            (q),
            dict(
                length=sweeper.values,
                prob=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiLengthClassificationData) -> RabiResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}
    amplitudes = {}
    chi2 = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        raw_x = qubit_data.length
        min_x = np.min(raw_x)
        max_x = np.max(raw_x)
        y = qubit_data.prob
        x = (raw_x - min_x) / (max_x - min_x)

        pguess = rabi_initial_guess(x, y, "length", signal=False)

        try:
            popt, perr, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                sigma=qubit_data.error,
                signal=False,
                x_limits=(min_x, max_x),
            )
            durations[qubit] = [pi_pulse_parameter, perr[2] * (max_x - min_x) / 2]
            fitted_parameters[qubit] = popt
            amplitudes = {key: [value, 0] for key, value in data.amplitudes.items()}
            chi2[qubit] = [
                chi2_reduced(
                    y,
                    rabi_length_function(raw_x, *popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiResults(
        drive_lines=data.drive_lines,
        length=durations,
        amplitude=amplitudes,
        fitted_parameters=fitted_parameters,
        rx90=data.rx90,
        chi2=chi2,
    )


def _plot(
    data: RabiLengthClassificationData,
    target: QubitId,
    fit: RabiResults | None = None,
):
    """Plotting function for RabiLength classification experiment."""
    return plot_probabilities(data, target, fit, data.rx90)


rabi_length = Protocol(_acquisition, _fit, _plot, update_rabi_parameters)
"""RabiLength Routine object.

In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
to find the drive pulse length that creates a rotation of a desired angle.
"""
