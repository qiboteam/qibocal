from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import chi2_reduced
from . import utils
from .parent_classes import (
    InputError,
    RabiLengthData,
    RabiLengthParameters,
    RabiLengthResults,
)

__all__ = ["rabi_length"]


RabiLenType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class ClassificationResults(RabiLengthResults):
    """RabiLength outputs."""

    chi2: dict[QubitId, float] = field(default_factory=dict)
    """Chi2 from each qubit's fit."""


@dataclass
class RabiLengthData(RabiLengthData):
    """RabiLength acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired for classification experiment."""


def _acquisition(
    params: RabiLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Classification Experiment.
    """

    drive_lines = params.drive_lines if params.drive_lines is not None else targets
    if len(drive_lines) != len(targets):
        raise InputError(
            "Each qubit has to be assigned to a drive line; "
            "If inserted, drive_lines must have the same length of targets list."
        )

    sequence, qd_pulses, delays, amplitudes, updates = utils.sequence_length(
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

    data = RabiLengthData(amplitudes=amplitudes, rx90=params.rx90)

    # execute the sweep
    results = platform.execute(
        [sequence],
        [[sweeper]],
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
            RabiLenType,
            (q),
            dict(
                length=sweeper.values,
                prob=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
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

        pguess = utils.rabi_initial_guess(x, y, "length", signal=False)

        try:
            popt, perr, pi_pulse_parameter = utils.fit_length_function(
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
                    utils.rabi_length_function(raw_x, *popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return ClassificationResults(
        length=durations,
        amplitude=amplitudes,
        fitted_parameters=fitted_parameters,
        rx90=data.rx90,
        chi2=chi2,
    )


def _update(
    results: ClassificationResults, platform: CalibrationPlatform, target: QubitId
):
    update.drive_duration(results.length[target], results.rx90, platform, target)
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)


def _plot(data: RabiLengthData, fit: ClassificationResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot_probabilities(data, target, fit, data.rx90)


<<<<<<< HEAD
rabi_length = Protocol(_acquisition, _fit, _plot, _update)
"""RabiLength Protocol object."""
=======
rabi_length = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object.

In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
to find the drive pulse length that creates a rotation of a desired angle.
"""
>>>>>>> 1aaefb18a (refactor rabi)
