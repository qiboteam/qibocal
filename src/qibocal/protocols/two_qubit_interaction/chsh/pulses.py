"""Auxialiary functions to run CHSH using pulses."""

import numpy as np
from qibolab import PulseSequence

from .utils import READOUT_BASIS


def create_bell_sequence(platform, qubits, theta=np.pi / 4, bell_state=0):
    """Creates the pulse sequence to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    """
    natives0 = platform.natives.single_qubit[qubits[0]]
    natives1 = platform.natives.single_qubit[qubits[1]]

    sequence = PulseSequence()
    sequence += natives0.R(theta=np.pi / 2, phi=np.pi / 2)
    sequence += natives1.R(theta=np.pi / 2, phi=np.pi / 2)

    cz_sequence = platform.natives.two_qubit[qubits].CZ()
    sequence |= cz_sequence[:1]
    phases = {ch.split("/")[0]: vz.phase for ch, vz in cz_sequence[1:]}
    # sequence |= cz_sequence
    # phases = {ch.split("/")[0]: 0 for ch, vz in cz_sequence[1:]}

    sequence_after = natives1.R(theta=np.pi / 2, phi=phases[qubits[1]] - np.pi / 2)
    
    if bell_state == 0:
        phases[qubits[0]] += np.pi
    elif bell_state == 1:
        phases[qubits[0]] += 0
    elif bell_state == 2:
        phases[qubits[0]] += 0
        sequence_after += natives0.R(theta=np.pi, phi=phases[qubits[0]])
    elif bell_state == 3:
        phases[qubits[0]] += np.pi
        sequence_after += natives0.R(theta=np.pi, phi=phases[qubits[0]])

    sequence_after += natives0.R(theta=np.pi / 2, phi=phases[qubits[0]])

    phases[qubits[0]] += theta
    sequence_after += natives0.R(theta=np.pi / 2, phi=phases[qubits[0]] + np.pi)

    return sequence | sequence_after, phases


def create_chsh_sequences(
    platform, qubits, theta=np.pi / 4, bell_state=0, readout_basis=READOUT_BASIS
):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    chsh_sequences = {}
    ro_pulses = {}

    for basis in readout_basis:
        sequence, phases = create_bell_sequence(platform, qubits, theta, bell_state)
        measurements = PulseSequence()
        ro_pulses[basis] = {}
        for i, base in enumerate(basis):
            natives = platform.natives.single_qubit[qubits[i]]
            if base == "X":
                sequence += natives.R(theta=np.pi / 2, phi=phases[qubits[i]] + np.pi / 2)

            measurement_seq = natives.MZ()
            ro_pulses[basis][qubits[i]] = measurement_seq[0][1]
            measurements += measurement_seq

        chsh_sequences[basis] = sequence | measurements

    return chsh_sequences, ro_pulses
