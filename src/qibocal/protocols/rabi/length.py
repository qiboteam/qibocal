from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Parameters, QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import chi2_reduced, fallback_period, guess_period
from . import utils
from .length_signal import RabiLengthSignalData, RabiLengthSignalResults

__all__ = ["rabi_length"]


@dataclass
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""


@dataclass
class RabiLengthResults(RabiLengthSignalResults):
    """RabiLength outputs."""

    chi2: dict[QubitId, list[float]] = field(default_factory=dict)


RabiLenType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(RabiLengthSignalData):
    """RabiLength acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiLengthParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, delays, ro_pulses, amplitudes = utils.sequence_length(
        targets, params, platform, params.rx90, use_align=params.interpolated_sweeper
    )
    sweep_range = (
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    if params.interpolated_sweeper:
        sweeper = Sweeper(
            parameter=Parameter.duration_interpolated,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets],
        )
    else:
        sweeper = Sweeper(
            parameter=Parameter.duration,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets] + [delays[q] for q in targets],
        )

    data = RabiLengthData(amplitudes=amplitudes, rx90=params.rx90)

    # execute the sweep
    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for q in targets:
        prob = probability(results[ro_pulses[q].id], state=1)
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

        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, period, 0, 0]

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

    return RabiLengthResults(durations, amplitudes, fitted_parameters, data.rx90, chi2)


def _update(results: RabiLengthResults, platform: CalibrationPlatform, target: QubitId):
    update.drive_duration(results.length[target], results.rx90, platform, target)
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)


def _plot(data: RabiLengthData, fit: RabiLengthResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot_probabilities(data, target, fit, data.rx90)


rabi_length = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
