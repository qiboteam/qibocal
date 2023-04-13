from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import fitting, plot, rabi


@dataclass
class RabiLengthParameters(Parameters):
    pulse_duration_start: float
    pulse_duration_end: float
    pulse_duration_step: float
    nshots: int
    relaxation_time: float
    software_averages: int = 1


@dataclass
class RabiLengthResults(Results):
    length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    fitted_parameters: Dict[List[Tuple], List]


class RabiLengthData(DataUnits):
    def __init__(self):
        super().__init__(
            name="data",
            quantities={"time": "ns"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: RabiLengthParameters
) -> RabiLengthData:
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_duration_start (int): Initial drive pulse length for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse length for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse length for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **time[ns]**: Drive pulse duration in ns
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_duration**: pi pulse duration
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation length
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
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
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

    for iteration in range(params.software_averages):
        # sweep the parameter
        for duration in qd_pulse_duration_range:
            for qubit in qubits:
                qd_pulses[qubit].duration = duration
                ro_pulses[qubit].start = duration

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence, nshots=params.nshots)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].average.raw
                r.update(
                    {
                        "time[ns]": duration,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                        "resonator_type": platform.resonator_type,
                    }
                )
                data.add(r)
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    return RabiLengthResults(*fitting(data, "length"))


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    return plot(data, fit, qubit, "length")


rabi_length = Routine(_acquisition, _fit, _plot)
