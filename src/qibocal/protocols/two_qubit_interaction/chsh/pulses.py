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

    sequence |= platform.natives.two_qubit[qubits].CZ()

    sequence_after = natives1.R(theta=np.pi / 2, phi=-np.pi / 2)

    if bell_state == 0:
        phase = np.pi
    elif bell_state == 1:
        phase = 0
    elif bell_state == 2:
        phase = 0
        sequence_after += natives0.R(theta=np.pi, phi=phase)
    elif bell_state == 3:
        phase = np.pi
        sequence_after += natives0.R(theta=np.pi, phi=phase)

    sequence_after += natives0.R(theta=np.pi / 2, phi=phase)

    phase += theta
    sequence_after += natives0.R(theta=np.pi / 2, phi=phase + np.pi)

    return sequence | sequence_after, phase


def create_chsh_sequences(
    platform, qubits, theta=np.pi / 4, bell_state=0, readout_basis=READOUT_BASIS
):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    chsh_sequences = {}
    for basis in readout_basis:
        sequence, phase = create_bell_sequence(platform, qubits, theta, bell_state)
        measurements = PulseSequence()
        for i, base in enumerate(basis):
            natives = platform.natives.single_qubit[qubits[i]]
            if base == "X":
                sequence += natives.R(theta=np.pi / 2, phi=phase * (1 - i) + np.pi / 2)

            measurement_seq = natives.MZ()
            measurements += measurement_seq

        chsh_sequences[basis] = sequence | measurements

    return chsh_sequences
