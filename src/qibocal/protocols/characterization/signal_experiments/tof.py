from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.protocols.characterization.signal_experiments import utils

from .utils import signals

"""
Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
for calibrate the ground state and the excited state of the oscillator.
The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
Args:
    platform (Platform): custom abstract platform on which we perform the calibration.
    qubits (dict): Dict of target Qubit objects to perform the action
    nshots (int): number of times the pulse sequence will be repeated.
    relaxation_time (float): #For data processing nothing qubit related
Returns:
    A Data object with the raw data obtained for the fast and precision sweeps with the following keys
        - **MSR[V]**: Signal voltage mesurement in volts before demodulation
        - **iteration[dimensionless]**: Execution number
        - **qubit**: The qubit being tested
        - **iteration**: The iteration number of the many determined by software_averages
"""


@dataclass
class ToFParameters(Parameters):
    """ToF runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ToFResults(Results):
    """ToF outputs."""

    ToF: Dict[Union[str, int], float] = field(metadata=dict(update="time_of_flight"))
    """Time of flight"""


class ToFData(DataUnits):
    """ToF acquisition outputs."""

    def __init__(self):
        super().__init__(
            f"data",
            # f"data_q{qubit}_{state}",
            options=["qubit", "sample", "state"],
        )


def _acquisition(params: ToFParameters, platform: Platform, qubits: Qubits) -> ToFData:
    """Data acquisition for time of flight experiment."""

    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    data = ToFData()

    # retrieve and store the results for every qubit
    for qubit in qubits:
        r = state0_results[ro_pulses[qubit].serial].serialize
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [qubit] * number_of_samples,
                "sample": np.arange(number_of_samples),
                "state": [0] * number_of_samples,
            }
        )
        data.add_data_from_dict(r)

    # # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        r = state1_results[ro_pulses[qubit].serial].serialize
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [qubit] * number_of_samples,
                "sample": np.arange(number_of_samples),
                "state": [1] * number_of_samples,
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data
    return data


def _fit(data: ToFData) -> ToFResults:
    """Post-processing function for ToF."""
    return ToFResults({})


def _plot(data: ToFData, fit: ToFResults, qubit):
    """Plotting function for ResonatorSpectroscopy."""
    return signals(data, fit, qubit)


tof = Routine(_acquisition, _fit, _plot)
"""ResonatorSpectroscopy Routine object."""
