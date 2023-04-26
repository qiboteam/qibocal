import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import calibrate_qubit_states_fit


@plot("Qubit States", plots.qubit_states)
def calibrate_qubit_states(
    platform: AbstractPlatform,
    qubits: dict,
    nshots,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
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
    data = DataUnits(name="data", options=["qubit", "iteration", "state"])

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state0_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * nshots,
                "iteration": np.arange(nshots),
                "state": [0] * nshots,
            }
        )
        data.add_data_from_dict(r)

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state1_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * nshots,
                "iteration": np.arange(nshots),
                "state": [1] * nshots,
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
    yield calibrate_qubit_states_fit(
        data, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
    )
