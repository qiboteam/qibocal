"""Testing update_* helper functions. """

import random
import re

import pytest
from qibolab import create_platform
from qibolab.native import VirtualZPulse
from qibolab.pulses import Drag

from qibocal import update

PLATFORM = create_platform("dummy")
QUBITS = list(PLATFORM.qubits.values())
PAIRS = list(PLATFORM.pairs)
RANDOM_FLOAT = random.random()
RANDOM_INT = random.randint(0, 10)


def generate_update_list(length):
    return [random.random()] * length


@pytest.mark.parametrize("qubit", QUBITS)
def test_readout_frequency_update(qubit):
    update.readout_frequency(RANDOM_INT, PLATFORM, qubit.name)
    assert qubit.native_gates.MZ.frequency == RANDOM_INT
    if qubit.native_gates.MZ.if_frequency is not None:
        assert qubit.readout_frequency == RANDOM_INT
        assert (
            qubit.native_gates.MZ.if_frequency == RANDOM_INT - qubit.readout_frequency
        )


@pytest.mark.parametrize("qubit", QUBITS)
def test_update_bare_resonator_frequency_update(qubit):
    update.bare_resonator_frequency(RANDOM_INT, PLATFORM, qubit.name)
    assert qubit.bare_resonator_frequency == RANDOM_INT


@pytest.mark.parametrize("qubit", QUBITS)
def test_readout_amplitude_update(qubit):
    update.readout_amplitude(RANDOM_FLOAT, PLATFORM, qubit.name)
    assert qubit.native_gates.MZ.amplitude == RANDOM_FLOAT


@pytest.mark.parametrize("qubit", QUBITS)
def test_readout_attenuation_update(qubit):
    update.readout_attenuation(RANDOM_INT, PLATFORM, qubit.name)
    assert qubit.readout.attenuation == RANDOM_INT


@pytest.mark.parametrize("qubit", QUBITS)
def test_drive_frequency_update(qubit):
    update.drive_frequency(RANDOM_INT, PLATFORM, qubit.name)
    assert qubit.native_gates.RX.frequency == RANDOM_INT
    assert qubit.drive_frequency == RANDOM_INT


@pytest.mark.parametrize("qubit", QUBITS)
def test_drive_amplitude_update(qubit):
    update.drive_amplitude(RANDOM_FLOAT, PLATFORM, qubit.name)
    assert qubit.native_gates.RX.amplitude == RANDOM_FLOAT


@pytest.mark.parametrize("qubit", QUBITS)
def test_classification_update(qubit):
    # generate random lists
    mean_gnd_state = generate_update_list(2)
    mean_exc_state = generate_update_list(2)
    # perform update
    update.iq_angle(RANDOM_FLOAT, PLATFORM, qubit.name)
    update.threshold(RANDOM_FLOAT, PLATFORM, qubit.name)
    update.mean_gnd_states(mean_gnd_state, PLATFORM, qubit.name)
    update.mean_exc_states(mean_exc_state, PLATFORM, qubit.name)
    update.readout_fidelity(RANDOM_FLOAT, PLATFORM, qubit.name)
    update.assignment_fidelity(RANDOM_FLOAT, PLATFORM, qubit.name)

    # assert
    assert qubit.iq_angle == RANDOM_FLOAT
    assert qubit.threshold == RANDOM_FLOAT
    assert qubit.mean_gnd_states == mean_gnd_state
    assert qubit.mean_exc_states == mean_exc_state
    assert qubit.readout_fidelity == RANDOM_FLOAT
    assert qubit.assignment_fidelity == RANDOM_FLOAT


@pytest.mark.parametrize("pair", PAIRS)
def test_virtual_phases_update(pair):
    if PLATFORM.pairs[pair].native_gates.CZ is not None:
        results = {qubit: RANDOM_FLOAT for qubit in pair}

        update.virtual_phases(results, PLATFORM, pair)
        if PLATFORM.pairs[pair].native_gates.CZ is not None:
            for pulse in PLATFORM.pairs[pair].native_gates.CZ.pulses:
                if isinstance(pulse, VirtualZPulse):
                    assert pulse == VirtualZPulse(
                        qubit=pulse.qubit, phase=results[pulse.qubit.name]
                    )


@pytest.mark.parametrize("pair", PAIRS)
def test_CZ_params_update(pair):
    if hasattr(PLATFORM.pairs[pair].native_gates, "CZ"):
        if PLATFORM.pairs[pair].native_gates.CZ is not None:
            update.CZ_amplitude(RANDOM_FLOAT, PLATFORM, pair)
            update.CZ_duration(RANDOM_INT, PLATFORM, pair)

            for pulse in PLATFORM.pairs[pair].native_gates.CZ.pulses:
                if pulse.qubit.name == pair[1]:
                    assert pulse.duration == RANDOM_INT
                    assert pulse.amplitude == RANDOM_FLOAT


@pytest.mark.parametrize("qubit", QUBITS)
def drive_duration_update(qubit):
    update.readout_amplitude(RANDOM_INT, PLATFORM, qubit.name)
    assert qubit.native_gates.RX.duration == RANDOM_FLOAT


@pytest.mark.parametrize("qubit", QUBITS)
def test_coherence_params_update(qubit):
    update.t1(RANDOM_INT, PLATFORM, qubit.name)
    update.t2(RANDOM_INT, PLATFORM, qubit.name)
    update.t2_spin_echo(RANDOM_INT, PLATFORM, qubit.name)

    assert qubit.t1 == RANDOM_INT
    assert qubit.t2 == RANDOM_INT
    assert qubit.t2_spin_echo == RANDOM_INT


@pytest.mark.parametrize("qubit", QUBITS)
def test_drag_pulse_beta_update(qubit):
    update.drag_pulse_beta(RANDOM_FLOAT, PLATFORM, qubit.name)

    rel_sigma = re.findall(
        r"[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+", qubit.native_gates.RX.shape
    )[0]
    assert qubit.native_gates.RX.shape == repr(Drag(rel_sigma, RANDOM_FLOAT))


@pytest.mark.parametrize("qubit", QUBITS)
def test_sweetspot_update(qubit):
    update.sweetspot(RANDOM_FLOAT, PLATFORM, qubit.name)
    assert qubit.sweetspot == RANDOM_FLOAT


# FIXME: missing qubit 4 RX12
@pytest.mark.parametrize("qubit", QUBITS[:-1])
def test_12_transition_update(qubit):
    update.drive_12_amplitude(RANDOM_FLOAT, PLATFORM, qubit.name)
    update.frequency_12_transition(RANDOM_INT, PLATFORM, qubit.name)
    update.anharmonicity(RANDOM_INT, PLATFORM, qubit.name)

    assert qubit.native_gates.RX12.amplitude == RANDOM_FLOAT
    assert qubit.native_gates.RX12.frequency == RANDOM_INT
    assert qubit.anharmonicity == RANDOM_INT


@pytest.mark.parametrize("qubit", QUBITS)
def test_twpa_update(qubit):
    update.twpa_frequency(RANDOM_INT, PLATFORM, qubit.name)
    update.twpa_power(RANDOM_FLOAT, PLATFORM, qubit.name)

    assert qubit.twpa.local_oscillator.frequency == RANDOM_INT
    assert qubit.twpa.local_oscillator.power == RANDOM_FLOAT
