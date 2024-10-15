from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from . import utils

PROJECTIONS = ['Z', 'Y', 'X']

@dataclass
class RabiLengthVoltParameters(Parameters):
    """RabiLengthVolt runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""


@dataclass
class RabiLengthVoltResults(Results):
    """RabiLengthVolt outputs."""

    length: dict[QubitId, tuple[int, Optional[float]]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, tuple[float, Optional[float]]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


RabiLenVoltType = np.dtype(
    [("length", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthVoltData(Data):
    """RabiLength acquisition outputs."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenVoltType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiLengthVoltParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthVoltData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in targets:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.pulse_duration_start
        )
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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

    data = RabiLengthVoltData(amplitudes=amplitudes)

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
    projection = PROJECTIONS[0]
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            RabiLenVoltType,
            (qubit, projection),
            dict(
                length=qd_pulse_duration_range,
                signal=result.magnitude,
                phase=result.phase,
            ),
        )
    return data


def _fit(data: RabiLengthVoltData) -> RabiLengthVoltResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[(qubit, PROJECTIONS[0])]
        rabi_parameter = qubit_data.length
        voltages = qubit_data.signal

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min) - 1 / 2

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=1)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5

        pguess = [0, np.sign(y[0]) * 0.5, 1 / f, 0, 0]
        try:
            popt, _ = curve_fit(
                utils.rabi_length_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, -1, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )
            translated_popt = [  # change it according to the fit function
                (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = (
                translated_popt[2]
                / 2
                * utils.period_correction_factor(phase=translated_popt[3])
            )
            durations[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = translated_popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthVoltResults(durations, data.amplitudes, fitted_parameters)


def _update(results: RabiLengthVoltResults, platform: Platform, target: QubitId):
    update.drive_duration(results.length[target], platform, target)


def _plot(data: RabiLengthVoltData, fit: RabiLengthVoltResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, target, fit)


rabi_length_signal = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
