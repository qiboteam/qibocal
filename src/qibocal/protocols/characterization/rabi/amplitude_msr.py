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
from qibocal.protocols.characterization.rabi.amplitude import (
    RabiAmplitudeData,
    RabiAmplitudeParameters,
    RabiAmplitudeResults,
)

from . import utils


@dataclass
class RabiAmplitudeVoltParameters(RabiAmplitudeParameters):
    """RabiAmplitude runcard inputs."""


@dataclass
class RabiAmplitudeVoltResults(RabiAmplitudeResults):
    """RabiAmplitude outputs."""


RabiAmpVoltType = np.dtype(
    [("amp", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeVoltData(RabiAmplitudeData):
    """RabiAmplitude data acquisition."""


def _acquisition(
    params: RabiAmplitudeVoltParameters, platform: Platform, qubits: Qubits
) -> RabiAmplitudeVoltData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.FACTOR,
    )

    data = RabiAmplitudeVoltData(durations=durations)

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
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            RabiAmpVoltType,
            (qubit),
            dict(
                amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
                msr=result.magnitude,
                phase=result.phase,
            ),
        )
    return data


def _fit(data: RabiAmplitudeVoltData) -> RabiAmplitudeVoltResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        rabi_parameter = qubit_data.amp
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
        local_maxima = find_peaks(mags, threshold=10)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5
        pguess = [0.5, 1, 1 / f, np.pi / 2]
        try:
            popt, _ = curve_fit(
                utils.rabi_amplitude_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
            )
            translated_popt = [  # Change it according to fit function changes
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / (x_max - x_min) / popt[2],
            ]
            pi_pulse_parameter = np.abs((translated_popt[2]) / 2)

        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            fitted_parameters = [0] * 4

        pi_pulse_amplitudes[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return RabiAmplitudeVoltResults(
        pi_pulse_amplitudes, data.durations, fitted_parameters
    )


def _plot(data: RabiAmplitudeVoltData, qubit, fit: RabiAmplitudeVoltResults = None):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, qubit, fit)


def _update(results: RabiAmplitudeVoltResults, platform: Platform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], platform, qubit)


rabi_amplitude_msr = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
