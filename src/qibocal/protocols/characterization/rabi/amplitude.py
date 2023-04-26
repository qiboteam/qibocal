from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

from .utils import plot, rabi_amplitude_fit


@dataclass
class RabiAmplitudeParameters(Parameters):
    min_amp_factor: float
    max_amp_factor: float
    step_amp_factor: float
    pulse_length: float
    nshots: int
    relaxation_time: float


@dataclass
class RabiAmplitudeResults(Results):
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    fitted_parameters: Dict[List[Tuple], List]


class RabiAmplitudeData(DataUnits):
    def __init__(self, resonator_type):
        super().__init__(
            "data",
            {"amplitude": "dimensionless", "length": "ns"},
            options=["qubit"],
        )
        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        return self._resonator_type


def _acquisition(
    params: RabiAmplitudeParameters, platform: AbstractPlatform, qubits: Qubits
) -> RabiAmplitudeData:
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **amplitude[dimensionless]**: Drive pulse amplitude
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_amplitude**: pi pulse amplitude
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit].duration = params.pulse_length
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
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData(platform.resonator_type)

    # sweep the parameter
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        r = result.raw
        r.update(
            {
                "amplitude[dimensionless]": qd_pulses[qubit].amplitude
                * qd_pulse_amplitude_range,
                "length[ns]": len(qd_pulse_amplitude_range)
                * [qd_pulses[qubit].duration],
                "qubit": len(qd_pulse_amplitude_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    qubits = data.df["qubit"].unique()

    pi_pulse_amplitudes = {}
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        rabi_parameter = qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude
        durations[qubit] = qubit_data["length"].pint.to("ns").pint.magnitude.unique()

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

        pguess = [0.5, 1, f, np.pi / 2]
        try:
            popt, pcov = curve_fit(rabi_amplitude_fit, x, y, p0=pguess, maxfev=100000)
            translated_popt = [
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] / (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / (x_max - x_min) * popt[2],
            ]
            pi_pulse_parameter = np.abs((1.0 / translated_popt[2]) / 2)
            pi_pulse_amplitudes[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = translated_popt
        except:
            log.warning("rabi_fit: the fitting was not succesful")

        return RabiAmplitudeResults(pi_pulse_amplitudes, durations, fitted_parameters)


def _plot(data: RabiAmplitudeData, fit: RabiAmplitudeResults, qubit):
    return plot(data, fit, qubit)


rabi_amplitude = Routine(_acquisition, _fit, _plot)
