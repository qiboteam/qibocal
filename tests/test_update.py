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
RANDOM_FLOAT = {qubit: random.random for qubit in PLATFORM.qubits}
RANDOM_INT = {qubit: random.randint(0, 10) for qubit in PLATFORM.qubits}


def generate_update_list(length):
    return {qubit: [random.random] * length for qubit in PLATFORM.qubits}


def test_readout_frequency_update():
    update.readout_frequency(FREQUENCIES_GHZ, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.MZ.frequency == FREQUENCIES_HZ[qubit_id]
        if qubit.native_gates.MZ.if_frequency is not None:
            assert qubit.readout_frequency == FREQUENCIES_HZ[qubit_id]
            assert (
                qubit.native_gates.MZ.if_frequency
                == FREQUENCIES_HZ[qubit_id] - qubit.readout_frequency
            )


def test_update_bare_resonator_frequency():
    update.bare_resonator_frequency(FREQUENCIES_GHZ, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.bare_resonator_frequency == FREQUENCIES_HZ[qubit_id]


def test_readout_amplitude_update():
    update.readout_amplitude(RANDOM_FLOAT, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.MZ.amplitude == RANDOM_FLOAT[qubit_id]


def test_readout_attenuation_update():
    readout_attenuation = 123
    update.readout_attenuation(
        {qubit: readout_attenuation for qubit in PLATFORM.qubits}, PLATFORM
    )
    for qubit in PLATFORM.qubits.values():
        assert qubit.readout.attenuation == readout_attenuation


def test_drive_frequency_update():
    update.drive_frequency(FREQUENCIES_GHZ, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.RX.frequency == FREQUENCIES_HZ[qubit_id]
        assert qubit.drive_frequency == FREQUENCIES_HZ[qubit_id]


def test_drive_amplitude_update():
    update.drive_amplitude(RANDOM_FLOAT, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.RX.amplitude == RANDOM_FLOAT[qubit_id]


def test_classification_update():
    # generate random lists
    mean_gnd_state = generate_update_list(2)
    mean_exc_state = generate_update_list(2)
    classifiers_hpars = generate_update_list(4)

    # perform update
    update.iq_angle(RANDOM_FLOAT, PLATFORM)
    update.threshold(RANDOM_FLOAT, PLATFORM)
    update.mean_gnd_states(mean_gnd_state, PLATFORM)
    update.mean_exc_states(mean_exc_state, PLATFORM)
    update.classifiers_hpars(classifiers_hpars, PLATFORM)

    # assert
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.iq_angle == RANDOM_FLOAT[qubit_id]
        assert qubit.threshold == RANDOM_FLOAT[qubit_id]
        assert qubit.mean_gnd_states == mean_gnd_state[qubit_id]
        assert qubit.mean_exc_states == mean_exc_state[qubit_id]
        assert qubit.classifiers_hpars == classifiers_hpars[qubit_id]
