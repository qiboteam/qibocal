from collections import defaultdict

import numpy as np
from qibolab.pulses import PulseSequence


def create_mermin_sequence(platform, qubits):
    """Creates the pulse sequence to generate the bell states and with a theta-measurement"""

    virtual_z_phases = defaultdict(int)
    sequence = PulseSequence()

    sequence.add(
        platform.create_RX90_pulse(qubits[0], start=0, relative_phase=np.pi / 2)
    )
    sequence.add(
        platform.create_RX90_pulse(qubits[1], start=0, relative_phase=np.pi / 2)
    )
    sequence.add(
        platform.create_RX90_pulse(qubits[2], start=0, relative_phase=np.pi / 2)
    )

    (cz_sequence1, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
        qubits[0:2], sequence.finish
    )
    sequence.add(cz_sequence1)
    for qubit in cz_virtual_z_phases:
        virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    (cz_sequence2, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
        qubits[1:3], sequence.finish + 16  # why 16
    )
    sequence.add(cz_sequence2)
    for qubit in cz_virtual_z_phases:
        virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    t = sequence.finish + 16

    sequence.add(
        platform.create_RX90_pulse(
            qubits[0],
            start=t,
            relative_phase=virtual_z_phases[qubits[0]] - np.pi / 2,
        )
    )

    sequence.add(
        platform.create_RX90_pulse(
            qubits[2],
            start=t,
            relative_phase=virtual_z_phases[qubits[2]] - np.pi / 2,
        )
    )

    virtual_z_phases[qubits[0]] -= np.pi / 2

    return sequence, virtual_z_phases


def create_mermin_sequences(platform, qubits, readout_basis):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    mermin_sequences = []

    for basis in readout_basis:
        sequence, virtual_z_phases = create_mermin_sequence(platform, qubits)
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
            if base == "Y":
                sequence.add(
                    platform.create_RX90_pulse(
                        qubits[i],
                        start=t,
                        relative_phase=virtual_z_phases[qubits[i]],
                    )
                )
        measurement_start = sequence.finish

        for qubit in qubits:
            MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
            sequence.add(MZ_pulse)
        mermin_sequences.append(sequence)

    return mermin_sequences
