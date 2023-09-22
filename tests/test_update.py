"""Testing update_* helper functions. """
import random
import re

from qibolab import create_platform
from qibolab.native import VirtualZPulse
from qibolab.pulses import Drag

from qibocal import update
from qibocal.protocols.characterization.utils import GHZ_TO_HZ

PLATFORM = create_platform("dummy")
FREQUENCIES_GHZ = {qubit: random.randint(5, 9) for qubit in PLATFORM.qubits}
FREQUENCIES_HZ = {
    qubit: int(freq * GHZ_TO_HZ) for qubit, freq in FREQUENCIES_GHZ.items()
}
RANDOM_FLOAT = {qubit: random.random() for qubit in PLATFORM.qubits}
RANDOM_INT = {qubit: random.randint(0, 10) for qubit in PLATFORM.qubits}


def generate_update_list(length):
    return {qubit: [random.random()] * length for qubit in PLATFORM.qubits}


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


def test_update_bare_resonator_frequency_update():
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


def test_virtual_phases_update():
    results = {
        pair_name: {qubit: RANDOM_FLOAT for qubit in pair_name}
        for pair_name, pair in PLATFORM.pairs.items()
        if pair.native_gates.CZ is not None
    }
    update.virtual_phases(results, PLATFORM)
    for name, pair in PLATFORM.pairs.items():
        if pair.native_gates.CZ is not None:
            for pulse in pair.native_gates.CZ.pulses:
                if isinstance(pulse, VirtualZPulse):
                    assert pulse == VirtualZPulse(
                        qubit=pulse.qubit, phase=results[name][pulse.qubit.name]
                    )


def test_CZ_params_update():
    amplitudes = {
        name: random.random()
        for name, pair in PLATFORM.pairs.items()
        if pair.native_gates.CZ is not None
    }
    durations = {
        name: random.randint(0, 10)
        for name, pair in PLATFORM.pairs.items()
        if pair.native_gates.CZ is not None
    }

    update.CZ_amplitude(amplitudes, PLATFORM)
    update.CZ_duration(durations, PLATFORM)

    for name, pair in PLATFORM.pairs.items():
        if pair.native_gates.CZ is not None:
            for pulse in pair.native_gates.CZ.pulses:
                if pulse.qubit.name == name[1]:
                    assert pulse.duration == durations[name]
                    assert pulse.amplitude == amplitudes[name]


def drive_duration_update():
    update.readout_amplitude(RANDOM_INT, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.native_gates.RX.duration == RANDOM_FLOAT[qubit_id]


def test_coherence_params_update():
    update.t1(RANDOM_INT, PLATFORM)
    update.t2(RANDOM_INT, PLATFORM)
    update.t2_spin_echo(RANDOM_INT, PLATFORM)

    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.t1 == RANDOM_INT[qubit_id]
        assert qubit.t2 == RANDOM_INT[qubit_id]
        assert qubit.t2_spin_echo == RANDOM_INT[qubit_id]


def test_drag_pulse_beta_update():
    update.drag_pulse_beta(RANDOM_FLOAT, PLATFORM)

    for qubit_id, qubit in PLATFORM.qubits.items():
        rel_sigma = re.findall(
            r"[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+", qubit.native_gates.RX.shape
        )[0]
        assert qubit.native_gates.RX.shape == repr(
            Drag(rel_sigma, RANDOM_FLOAT[qubit_id])
        )


def test_sweetspot_update():
    update.sweetspot(RANDOM_FLOAT, PLATFORM)
    for qubit_id, qubit in PLATFORM.qubits.items():
        assert qubit.sweetspot == RANDOM_FLOAT[qubit_id]
        if qubit.flux is not None:
            assert qubit.flux.offset == RANDOM_FLOAT[qubit_id]
