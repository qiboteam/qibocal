"""Testing update_* helper functions. """
import random

from qibolab import create_platform

from qibocal import update
from qibocal.protocols.characterization.utils import GHZ_TO_HZ

PLATFORM = create_platform("dummy")
FREQUENCIES_GHZ = {qubit: random.randint(5, 9) for qubit in PLATFORM.qubits}
FREQUENCIES_HZ = {
    qubit: int(freq * GHZ_TO_HZ) for qubit, freq in FREQUENCIES_GHZ.items()
}
AMPLITUDES = {qubit: random.random for qubit in PLATFORM.qubits}


def test_readout_frequency_update():
    update.update_readout_frequency(FREQUENCIES_GHZ, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.MZ.frequency == FREQUENCIES_HZ[qubit_id]
        if qubit.native_gates.MZ.if_frequency is not None:
            assert qubit.readout_frequency == FREQUENCIES_HZ[qubit_id]
            assert (
                qubit.native_gates.MZ.if_frequency
                == FREQUENCIES_HZ[qubit_id] - qubit.readout_frequency
            )


def test_update_bare_resonator_frequency():
    update.update_bare_resonator_frequency(FREQUENCIES_GHZ, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.bare_resonator_frequency == FREQUENCIES_HZ[qubit_id]


def test_readout_amplitude_update():
    update.update_readout_amplitude(AMPLITUDES, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.MZ.amplitude == AMPLITUDES[qubit_id]
