from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Qubits, Routine
from qibocal.config import log
from qibocal.protocols.characterization.rabi.length import (
    RabiLengthData,
    RabiLengthParameters,
    RabiLengthResults,
)

from . import utils


@dataclass
class RabiLengthVoltParameters(RabiLengthParameters):
    """RabiLength runcard inputs."""


@dataclass
class RabiLengthVoltResults(RabiLengthResults):
    """RabiLength outputs."""


RabiLenVoltType = np.dtype(
    [("length", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthVoltData(RabiLengthData):
    """RabiLength acquisition outputs."""


def _acquisition(
    params: RabiLengthVoltParameters, platform: Platform, qubits: Qubits
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
    for qubit in qubits:
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
        [qd_pulses[qubit] for qubit in qubits],
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

    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            RabiLenVoltType,
            (qubit),
            dict(
                length=qd_pulse_duration_range,
                msr=result.magnitude,
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
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages = qubit_data.msr

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=1)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5

        pguess = [0.5, 0.5, 1 / f, np.pi / 2, 0]
        try:
            popt, _ = curve_fit(
                utils.rabi_length_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )
            translated_popt = [  # change it according to the fit function
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = np.abs(translated_popt[2] / 2)
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            translated_popt = [0, 0, 1, 0, 0]

        durations[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return RabiLengthVoltResults(durations, data.amplitudes, fitted_parameters)


def _update(results: RabiLengthVoltResults, platform: Platform, qubit: QubitId):
    update.drive_duration(results.length[qubit], platform, qubit)


def _plot(data: RabiLengthVoltData, fit: RabiLengthVoltResults, qubit):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, qubit, fit)


rabi_length_msr = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
