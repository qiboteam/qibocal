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
        phi = np.pi
    elif bell_state == 1:
        phi = 0
    elif bell_state == 2:
        phi = 0
        sequence_after += natives0.R(theta=np.pi, phi=phi)
    elif bell_state == 3:
        phi = np.pi
        sequence_after += natives0.R(theta=np.pi, phi=phi)

    sequence_after += natives0.R(theta=np.pi / 2, phi=phi)

    phi += theta
    sequence_after += natives0.R(theta=np.pi / 2, phi=phi + np.pi)

    return sequence | sequence_after


def create_chsh_sequences(
    platform, qubits, theta=np.pi / 4, bell_state=0, readout_basis=READOUT_BASIS
):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    chsh_sequences = {}
    ro_pulses = {}

    for basis in readout_basis:
        sequence = create_bell_sequence(platform, qubits, theta, bell_state)
        measurements = PulseSequence()
        ro_pulses[basis] = {}
        for i, base in enumerate(basis):
            natives = platform.natives.single_qubit[qubits[i]]
            if base == "X":
                sequence += natives.R(theta=np.pi / 2, phi=np.pi / 2)

            measurement_seq = natives.MZ()
            ro_pulses[basis][qubits[i]] = measurement_seq[0][1]
            measurements += measurement_seq

        chsh_sequences[basis] = sequence | measurements

    return chsh_sequences, ro_pulses
