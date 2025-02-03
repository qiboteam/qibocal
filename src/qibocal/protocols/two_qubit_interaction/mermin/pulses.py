from collections import defaultdict

import numpy as np
from qibolab.pulses import PulseSequence


def create_mermin_sequence(platform, qubits, theta=None):
    """Creates the pulse sequence to generate the bell states and with a theta-measurement"""

    nqubits = len(qubits)
    if theta is None:
        theta = ((nqubits - 1) * 0.25 * np.pi) % (2 * np.pi)

    virtual_z_phases = defaultdict(int)
    sequence = PulseSequence()

    for qubit in qubits:
        sequence.add(
            platform.create_RX90_pulse(
                qubit, start=0, relative_phase=virtual_z_phases[qubit] + np.pi / 2
            )
        )

    # TODO: Not hardcode topology

    # qubits[0] needs to be the center qubit where everything is connected
    for i in range(1, len(qubits)):
        (cz_sequence1, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
            [qubits[0]] + [qubits[i]], sequence.finish + 8  # TODO: ask for the 8
        )
        sequence.add(cz_sequence1)
        for qubit in cz_virtual_z_phases:
            virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    t = sequence.finish + 8

    for i in range(1, len(qubits)):
        sequence.add(
            platform.create_RX90_pulse(
                qubits[i],
                start=t,
                relative_phase=virtual_z_phases[qubits[i]] - np.pi / 2,
            )
        )

    virtual_z_phases[qubits[0]] -= theta

    return sequence, virtual_z_phases


def create_mermin_sequences(platform, qubits, readout_basis, theta):
    """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

    mermin_sequences = {}

    for basis in readout_basis:
        sequence, virtual_z_phases = create_mermin_sequence(
            platform, qubits, theta=theta
        )
        # t = sequence.finish
        for i, base in enumerate(basis):
            if base == "X":
                sequence.add(
                    platform.create_RX90_pulse(
                        qubits[i],
                        start=sequence.finish,
                        relative_phase=virtual_z_phases[qubits[i]] + np.pi / 2,
                    )
                )
            if base == "Y":
                sequence.add(
                    platform.create_RX90_pulse(
                        qubits[i],
                        start=sequence.finish,
                        relative_phase=virtual_z_phases[qubits[i]],
                    )
                )
        measurement_start = sequence.finish

        for qubit in qubits:
            sequence.add(platform.create_MZ_pulse(qubit, start=measurement_start))

        mermin_sequences[basis] = sequence
    return mermin_sequences
