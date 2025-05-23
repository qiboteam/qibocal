from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import fallback_period, guess_period, readout_frequency
from qibocal.result import magnitude, phase

from . import utils

__all__ = ["rabi_length_signal", "RabiLengthSignalResults"]


@dataclass
class RabiLengthSignalParameters(Parameters):
    """RabiLengthSignal runcard inputs."""

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
class RabiLengthSignalResults(Results):
    """RabiLengthSignal outputs."""

    length: dict[QubitId, Union[int, list[float]]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, Union[float, list[float]]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    rx90: bool
    """Pi or Pi_half calibration"""


RabiLenSignalType = np.dtype(
    [("length", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthSignalData(Data):
    """RabiLength acquisition outputs."""

    rx90: bool
    """Pi or Pi_half calibration"""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiLengthSignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthSignalData:
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

    data = RabiLengthSignalData(amplitudes=amplitudes, rx90=params.rx90)

    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for q in targets:
        result = results[ro_pulses[q].id]
        data.register_qubit(
            RabiLenSignalType,
            (q),
            dict(
                length=sweeper.values,
                signal=magnitude(result),
                phase=phase(result),
            ),
        )
    return data


def _fit(data: RabiLengthSignalData) -> RabiLengthSignalResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages = qubit_data.signal

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min) - 1 / 2

        period = fallback_period(guess_period(x, y))
        pguess = [0, np.sign(y[0]) * 0.5, period, 0, 0]
        try:
            popt, _, pi_pulse_parameter = utils.fit_length_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            durations[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthSignalResults(
        durations, data.amplitudes, fitted_parameters, data.rx90
    )


def _update(
    results: RabiLengthSignalResults, platform: CalibrationPlatform, target: QubitId
):
    update.drive_duration(results.length[target], results.rx90, platform, target)
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)


def _plot(data: RabiLengthSignalData, fit: RabiLengthSignalResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, target, fit, data.rx90)


rabi_length_signal = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
