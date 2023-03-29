import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import AcquisitionType, AveragingMode
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("Fast Reset", plots.fast_reset_states)
def fast_reset_MSR(
    platform: AbstractPlatform,
    qubits: dict,
    nshots,
    fast_reset,
    relaxation_time=None,
):
    """
    Method which implements the pi pulse fine tuning for a given qubit by running a increasing number of pairs pi-pulses to check how far
    we shift from the expected ground state as the number of pairs increasing. This allows us to detect small detunings on our pi-pulse.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        number_of_pairs (int): Maximun number of pairs of pi-pulses (Maybe is better to input just the number or list of numbers)
        nshots (int) : Number of executions on hardware of the routine for averaging results
        fast_reset (bool) : Do fast reset or no
        relaxation_time (float): Wait time for the qubit to decohere back to the `gnd` state

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Signal voltage mesurement in volts before demodulation
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The SINGLE software iteration number

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create sequences of pulses:
    # sequence_n: RX - MZ

    # create a DataUnits object to store the results
    data = DataUnits(name="data", options=["qubit", "iteration", "n"])

    RX_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        # RX_pulses[qubit].amplitude = .25
        sequence.add(RX_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        sequence.add(ro_pulses[qubit])

    # execute the pulse sequence
    results = platform.execute_pulse_sequence(
        sequence,
        nshots=nshots,
        relaxation_time=relaxation_time,
        averaging_mode=AveragingMode.SINGLESHOT,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        fast_reset=fast_reset,
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = results[ro_pulse.serial].to_dict(average=False)
        r.update(
            {
                "qubit": [ro_pulse.qubit] * nshots,
                "iteration": np.arange(nshots),
                "n": [1] * nshots,
            }
        )
        data.add_data_from_dict(r)

    # Is it useful to inspect state 0 ?
    # ro_pulses = {}
    # sequence = PulseSequence()
    # for qubit in qubits:
    #     ro_pulses[qubit] = platform.create_qubit_readout_pulse(
    #         qubit, start=0
    #     )
    #     sequence.add(ro_pulses[qubit])

    # # execute the pulse sequence
    # results = platform.execute_pulse_sequence(
    #     sequence,
    #     nshots=nshots,
    #     relaxation_time=relaxation_time,
    # averaging_mode= AveragingMode.SINGLESHOT,
    # acquisition_type= AcquisitionType.DISCRIMINATION,,
    #     fast_reset=True,
    # )

    # # retrieve and store the results for every qubit
    # for ro_pulse in ro_pulses.values():
    #     r = results[ro_pulse.serial].to_dict(average=False)
    #     r.update(
    #         {
    #             "qubit": [ro_pulse.qubit] * nshots,
    #             "iteration": np.arange(nshots),
    #             "n": [0] * nshots,
    #         }
    #     )
    #     data.add_data_from_dict(r)

    yield data
