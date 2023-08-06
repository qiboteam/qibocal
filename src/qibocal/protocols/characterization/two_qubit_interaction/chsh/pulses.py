"""Auxialiary functions to run CHSH using pulses."""

from collections import defaultdict

import numpy as np
from qibolab.pulses import PulseSequence

from .utils import READOUT_BASIS


def create_bell_sequence(platform, qubits, theta=np.pi / 4, bell_state=0):
    """Creates the pulse sequence to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    """

    virtual_z_phases = defaultdict(int)

    sequence = PulseSequence()
    sequence.add(
        platform.create_RX90_pulse(qubits[0], start=0, relative_phase=np.pi / 2)
    )
    sequence.add(
        platform.create_RX90_pulse(qubits[1], start=0, relative_phase=np.pi / 2)
    )

    (cz_sequence, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
        qubits, sequence.finish
    )
    sequence.add(cz_sequence)
    for qubit in cz_virtual_z_phases:
        virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    t = sequence.finish

    sequence.add(
        platform.create_RX90_pulse(
            qubits[1],
            start=t,
            relative_phase=virtual_z_phases[qubits[1]] - np.pi / 2,
        )
    )

    if bell_state == 0:
        virtual_z_phases[qubits[0]] += np.pi
    elif bell_state == 1:
        virtual_z_phases[qubits[0]] += 0
    elif bell_state == 2:
        virtual_z_phases[qubits[0]] += 0
        sequence.add(
            platform.create_RX_pulse(
                qubits[0], start=t, relative_phase=virtual_z_phases[qubits[0]]
            )
        )
    elif bell_state == 3:
        virtual_z_phases[qubits[0]] += np.pi
        sequence.add(
            platform.create_RX_pulse(
                qubits[0], start=t, relative_phase=virtual_z_phases[qubits[0]]
            )
        )

    t = sequence.finish
    sequence.add(
        platform.create_RX90_pulse(
            qubits[0], start=t, relative_phase=virtual_z_phases[qubits[0]]
        )
    )
    virtual_z_phases[qubits[0]] += theta
    sequence.add(
        platform.create_RX90_pulse(
            qubits[0],
            start=sequence.finish,
            relative_phase=virtual_z_phases[qubits[0]] + np.pi,
        )
    )

    return sequence, virtual_z_phases


def create_chsh_sequences(
    platform, qubits, theta=np.pi / 4, bell_state=0, readout_basis=READOUT_BASIS
):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    chsh_sequences = {}

    for basis in readout_basis:
        sequence, virtual_z_phases = create_bell_sequence(
            platform, qubits, theta, bell_state
        )
        t = sequence.finish
        for i, base in enumerate(basis):
            if base == "X":
                sequence.add(
                    platform.create_RX90_pulse(
                        qubits[i],
                        start=t,
                        relative_phase=virtual_z_phases[qubits[i]] + np.pi / 2,
                    )
                )
        measurement_start = sequence.finish
        for qubit in qubits:
            MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
            sequence.add(MZ_pulse)
        chsh_sequences[basis] = sequence

    return chsh_sequences
