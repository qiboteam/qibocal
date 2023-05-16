from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log

from . import amplitude, utils


@dataclass
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration (ns)."""
    pulse_duration_end: float
    """Final pi pulse duration (ns)."""
    pulse_duration_step: float
    """Step pi pulse duration (ns)."""
    pulse_amplitude: float
    """Pi pulse amplitude. Same for all qubits."""
    nshots: int
    """Number of shots."""
    relaxation_time: float
    """Relxation time (ns)."""


@dataclass
class RabiLengthResults(Results):
    """RabiLength outputs."""

    length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    """Pi pulse duration for each qubit."""
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitting output."""


class RabiLengthData(amplitude.RabiAmplitudeData):
    """RabiLength acquisition outputs."""


def _acquisition(
    params: RabiLengthParameters, platform: AbstractPlatform, qubits: Qubits
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        qd_pulses[qubit].amplitude = params.pulse_amplitude

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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
    data = RabiLengthData()

    # sweep the parameter
    for duration in qd_pulse_duration_range:
        for qubit in qubits:
            print(duration)
            qd_pulses[qubit].duration = duration
            ro_pulses[qubit].start = qd_pulses[qubit].finish

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(sequence, nshots=params.nshots)

        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[qubit].average.raw
            r.update(
                {
                    "length[ns]": duration,
                    "amplitude[dimensionless]": float(qd_pulses[qubit].amplitude),
                    "qubit": qubit,
                }
            )
            data.add(r)

    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.df["qubit"].unique()
    pi_pulse_amplitudes = {}
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        rabi_parameter = qubit_data["length"].pint.to("ns").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude
        pi_pulse_amplitudes[qubit] = (
            qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
        )

        y_min = np.min(voltages.values)
        y_max = np.max(voltages.values)
        x_min = np.min(rabi_parameter.values)
        x_max = np.max(rabi_parameter.values)
        x = (rabi_parameter.values - x_min) / (x_max - x_min)
        y = (voltages.values - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
        f = x[index] / (x[1] - x[0])

        pguess = [1, 1, f, np.pi / 2, 0]

        try:
            popt, pcov = curve_fit(
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
            translated_popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - 2 * np.pi * x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = np.abs((1.0 / translated_popt[2]) / 2)
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            fitted_parameters = [0] * 4

        durations[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return RabiLengthResults(durations, pi_pulse_amplitudes, fitted_parameters)


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, fit, qubit)


rabi_length = Routine(_acquisition, _fit, _plot)
"""RabiLength Routine object."""
