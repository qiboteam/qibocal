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
class RabiAmplitudeSignalParameters(Parameters):
    """RabiAmplitude runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_length: Optional[float] = None
    """RX pulse duration [ns]."""


@dataclass
class RabiAmplitudeSignalResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, Union[float, list[float]]]
    """Drive amplitude for each qubit."""
    length: dict[QubitId, Union[float, list[float]]]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


RabiAmpSignalType = np.dtype(
    [("amp", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeSignalData(Data):
    """RabiAmplitudeSignal data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeSignalParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeSignalData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence, qd_pulses, ro_pulses, durations = utils.sequence_amplitude(
        targets, params, platform
    )

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.min_amp_factor,
        params.max_amp_factor,
        params.step_amp_factor,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.FACTOR,
    )

    data = RabiAmplitudeSignalData(durations=durations)

    # sweep the parameter
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
            RabiAmpSignalType,
            (qubit),
            dict(
                amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
                signal=result.magnitude,
                phase=result.phase,
            ),
        )
    return data


def _fit(data: RabiAmplitudeSignalData) -> RabiAmplitudeSignalResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        rabi_parameter = qubit_data.amp
        voltages = qubit_data.signal

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, period, np.pi]
        try:
            popt, _, pi_pulse_parameter = utils.fit_amplitude_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            pi_pulse_amplitudes[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeSignalResults(
        pi_pulse_amplitudes, data.durations, fitted_parameters
    )


def _plot(
    data: RabiAmplitudeSignalData,
    target: QubitId,
    fit: RabiAmplitudeSignalResults = None,
):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, target, fit)


def _update(results: RabiAmplitudeSignalResults, platform: Platform, target: QubitId):
    update.drive_amplitude(results.amplitude[target], platform, target)
    update.drive_duration(results.length[target], platform, target)


rabi_amplitude_signal = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
