from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

from .utils import fitting, plot, rabi


@dataclass
class RabiAmplitudeParameters(Parameters):
    pulse_amplitude_start: float
    pulse_amplitude_end: float
    pulse_amplitude_step: float
    nshots: int
    relaxation_time: float
    software_averages: int = 1


@dataclass
class RabiAmplitudeResults(Results):
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    fitted_parameters: Dict[List[Tuple], List]


class RabiAmplitudeData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"amplitude": "dimensionless"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: RabiAmplitudeParameters
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
        qd_pulses[qubit].duration = 40  # decided by Sergi
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.pulse_amplitude_start,
        params.pulse_amplitude_end,
        params.pulse_amplitude_step,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData()

    for iteration in range(params.software_averages):
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
            r = result.to_dict()
            r.update(
                {
                    "amplitude[dimensionless]": qd_pulse_amplitude_range,
                    "qubit": len(qd_pulse_amplitude_range) * [qubit],
                    "iteration": len(qd_pulse_amplitude_range) * [iteration],
                    "resonator_type": len(qd_pulse_amplitude_range)
                    * [platform.resonator_type],
                }
            )
            data.add_data_from_dict(r)
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    return RabiAmplitudeResults(*fitting(data, "amplitude"))


def _plot(data: RabiAmplitudeData, fit: RabiAmplitudeResults, qubit):
    return plot(data, fit, qubit, "amplitude")


rabi_amplitude = Routine(_acquisition, _fit, _plot)
