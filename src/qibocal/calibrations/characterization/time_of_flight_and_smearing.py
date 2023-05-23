import numpy as np
from qibolab.executionparameters import AcquisitionType, AveragingMode
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot


@plot("Qubit States", plots.signals)
def time_of_flight(
    platform: AbstractPlatform,
    qubits: dict,
    nshots,
    relaxation_time=50e-9,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
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

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # TODO: Check if you can extract better information from any of this
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

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

    # create a DataUnits object to store the results
    data = DataUnits(name="data", options=["qubit", "sample", "state"])

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence,
        nshots=nshots,
        relaxation_time=relaxation_time,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve and store the results for every qubit
    # TODO: Something with iteration
    for ro_pulse in ro_pulses.values():
        r = state0_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                "sample": np.arange(len(r["MSR[V]"])),
                "state": [0] * len(r["MSR[V]"]),
            }
        )
        data.add_data_from_dict(r)

    # # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        nshots=nshots,
        relaxation_time=relaxation_time,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state1_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                "sample": np.arange(len(r["MSR[V]"])),
                "state": [1] * len(r["MSR[V]"]),
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
