from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.utils import fallback_period, guess_period

from . import utils


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


@dataclass
class RabiLengthSignalResults(Results):
    """RabiLengthSignal outputs."""

    length: dict[QubitId, Union[int, list[float]]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, Union[float, list[float]]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


RabiLenSignalType = np.dtype(
    [("length", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthSignalData(Data):
    """RabiLength acquisition outputs."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiLengthSignalParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthSignalData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, ro_pulses, amplitudes = utils.sequence_length(
        targets, params, platform
    )

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    sweeper = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.ABSOLUTE,
    )
    data = RabiLengthSignalData(amplitudes=amplitudes)

    # execute the sweep
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            RabiLenSignalType,
            (qubit),
            dict(
                length=qd_pulse_duration_range,
                signal=result.magnitude,
                phase=result.phase,
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

    return RabiLengthSignalResults(durations, data.amplitudes, fitted_parameters)


def _update(results: RabiLengthSignalResults, platform: Platform, target: QubitId):
    update.drive_duration(results.length[target], platform, target)
    update.drive_amplitude(results.amplitude[target], platform, target)


def _plot(data: RabiLengthSignalData, fit: RabiLengthSignalResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, target, fit)


rabi_length_signal = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
