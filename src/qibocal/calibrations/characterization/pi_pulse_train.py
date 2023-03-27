import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("MSR vs Time", plots.time_msr_phase_train)
def pi_pulse_train_MSR(
    platform: AbstractPlatform,
    qubits: dict,
    number_of_pairs,
    nshots,
    relaxation_time=None,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        number_of_pairs (int): Maximun number of pairs of pi-pulses (Maybe is better to input just the number or list of numbers)
        nshots (int): number of times the pulse sequence will be repeated.

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Signal voltage mesurement in volts before demodulation
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create sequences of pulses:
    # sequence_n: RX*2n - MZ

    # TODO: I think there is a way to make this a sweep on gates.

    number_of_RX_pairs = np.arange(0, 2 * number_of_pairs, 2)

    # create a DataUnits object to store the results
    data = DataUnits(name="data", options=["qubit", "iteration", "n"])

    for n in number_of_RX_pairs:
        RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            if n > 0:
                RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                sequence.add(RX_pulses[qubit])
                for i in range(1, n):
                    RX_pulses[qubit] = platform.create_RX_pulse(
                        qubit, start=RX_pulses[qubit].duration * i
                    )
                    sequence.add(RX_pulses[qubit])

                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit, start=RX_pulses[qubit].finish
                )
            else:
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
            sequence.add(ro_pulses[qubit])

        # execute the first pulse sequence
        results = platform.execute_pulse_sequence(
            sequence, nshots=nshots, relaxation_time=relaxation_time
        )

        # retrieve and store the results for every qubit
        for ro_pulse in ro_pulses.values():
            r = results[ro_pulse.serial].to_dict(average=False)
            r.update(
                {
                    "qubit": ro_pulse.qubit,
                    "iteration": 0,  # Get rid of this and software averages. Need for the plotting
                    "n": n,
                }
            )
            data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
