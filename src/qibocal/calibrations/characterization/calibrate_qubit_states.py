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
    qubits: list,
    nshots,
    points=10,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubit (int): index representing the target qubit into the chip.
        niter (int): number of times the pulse sequence will be reproduced.
        points (int): every points step data are saved.

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number

    """

    platform.reload_settings()
    # create exc sequence
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].duration
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    data = DataUnits(name="data", options=["qubit", "iteration", "state"])

    state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)
    for qubit in qubits:
        msr, phase, i, q = state0_results["demodulated_integrated_binned"][
            ro_pulses[qubit].serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "qubit": [qubit] * nshots,
            "iteration": np.arange(nshots),
            "state": [0] * nshots,
        }
        data.add_data_from_dict(results)

    state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
    for qubit in qubits:
        msr, phase, i, q = state1_results["demodulated_integrated_binned"][
            ro_pulses[qubit].serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "qubit": [qubit] * nshots,
            "iteration": np.arange(nshots),
            "state": [1] * nshots,
        }
        data.add_data_from_dict(results)

    yield data
    yield calibrate_qubit_states_fit(data, nshots=nshots, qubits=qubits)
