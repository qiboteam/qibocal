import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibolab.pulses import PulseSequence
from utils import calculate_frequencies


class BellExperiment:
    def __init__(self, platform, nqubits, readout_error_model=(0.0, 0.0)):
        """Platform should be left None for simulation"""
        self.platform = platform
        self.nqubits = nqubits
        self.rerr = readout_error_model

    def create_bell_sequence(self, qubits, theta=np.pi / 4, bell_state=0):
        """Creates the pulse sequence to generate the bell states and with a theta-measurement
        bell_state chooses the initial bell state for the test:
        0 -> |00>+|11>
        1 -> |00>-|11>
        2 -> |10>-|01>
        3 -> |10>+|01>
        """
        platform = self.platform

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

    def create_chsh_sequences(self, qubits, theta=np.pi / 4, bell_state=0):
        """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

        platform = self.platform

        readout_basis = [["Z", "Z"], ["Z", "X"], ["X", "Z"], ["X", "X"]]

        chsh_sequences = []

        for basis in readout_basis:
            sequence, virtual_z_phases = self.create_bell_sequence(
                qubits, theta, bell_state
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
            chsh_sequences.append(sequence)

        return chsh_sequences

    def create_bell_circuit(self, qubits, theta=np.pi / 4, bell_state=0, native=True):
        """Creates the circuit to generate the bell states and with a theta-measurement
        bell_state chooses the initial bell state for the test:
        0 -> |00>+|11>
        1 -> |00>-|11>
        2 -> |10>-|01>
        3 -> |10>+|01>
        Native defaults to only using GPI2 and GPI gates.
        """
        nqubits = self.nqubits

        c = Circuit(nqubits)
        p = [0, 0]
        if native:
            c.add(gates.GPI2(qubits[0], np.pi / 2))
            c.add(gates.GPI2(qubits[1], np.pi / 2))
            c.add(gates.CZ(qubits[0], qubits[1]))
            c.add(gates.GPI2(qubits[1], -np.pi / 2))
            if bell_state == 0:
                p[0] += np.pi
            elif bell_state == 1:
                p[0] += 0
            elif bell_state == 2:
                p[0] += 0
                c.add(gates.GPI(qubits[0], p[0]))
            elif bell_state == 3:
                p[0] += np.pi
                c.add(gates.GPI(qubits[0], p[0]))

            c.add(gates.GPI2(qubits[0], p[0]))
            p += theta
            c.add(gates.GPI2(qubits[0], p[0] + np.pi))

        else:
            c.add(gates.H(qubits[0]))
            c.add(gates.H(qubits[1]))
            c.add(gates.CZ(qubits[0], qubits[1]))
            c.add(gates.H(qubits[1]))

            if bell_state == 1:
                c.add(gates.Z(qubits[0]))
            elif bell_state == 2:
                c.add(gates.Z(qubits[0]))
                c.add(gates.X(qubits[0]))
            elif bell_state == 3:
                c.add(gates.X(qubits[0]))

            c.add(gates.RY(qubits[0], theta))
        return c, p

    def create_chsh_circuits(
        self, qubits, theta=np.pi / 4, bell_state=0, native=True, rerr=None
    ):
        """Creates the circuits needed for the 4 measurement settings for chsh.
        Native defaults to only using GPI2 and GPI gates.
        rerr adds a readout bitflip error to the simulation.
        """
        if not rerr:
            rerr = self.rerr

        readout_basis = [["Z", "Z"], ["Z", "X"], ["X", "Z"], ["X", "X"]]

        chsh_circuits = []

        for basis in readout_basis:
            c, p = self.create_bell_circuit(qubits, theta, bell_state, native)
            for i, base in enumerate(basis):
                if base == "X":
                    if native:
                        c.add(gates.GPI2(qubits[i], p[i] + np.pi / 2))
                    else:
                        c.add(gates.H(qubits[i]))
            for qubit in qubits:
                c.add(gates.M(qubit, p0=rerr[0], p1=rerr[1]))
            chsh_circuits.add(c)

        return chsh_circuits

    def compute_chsh(self, frequencies, basis):
        """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
        chsh = 0
        aux = 0
        for freq in frequencies:
            for outcome in freq:
                if aux == 1 + 2 * (
                    basis % 2
                ):  # This value sets where the minus sign is in the CHSH inequality
                    chsh -= (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
                else:
                    chsh += (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
            aux += 1
        nshots = sum(freq[x] for x in freq)
        return chsh / nshots

    def plot(
        self,
        thetas,
        chsh_values,
        mitigated_chsh_values=None,
        exact_chsh_values=None,
        title="test",
    ):
        """Standard plot for the chsh results. It can plot the mitigated and exact expected values as well."""

        fig = plt.figure(figsize=(12, 8))
        plt.rcParams.update({"font.size": 22})
        plt.plot(thetas, chsh_values, "o-", label="bare")
        if mitigated_chsh_values:
            plt.plot(thetas, mitigated_chsh_values, "o-", label="mitigated")
        if exact_chsh_values:
            plt.plot(thetas, exact_chsh_values, "o-", label="exact")
        plt.grid(which="major", axis="both")
        plt.rcParams.update({"font.size": 16})
        plt.legend()
        plt.axhline(y=2, color="r", linestyle="-")
        plt.axhline(y=-2, color="r", linestyle="-")
        plt.axhline(y=np.sqrt(2) * 2, color="k", linestyle="-.")
        plt.axhline(y=-np.sqrt(2) * 2, color="k", linestyle="-.")
        plt.xlabel("Theta")
        plt.ylabel("CHSH value")
        plt.title(f"Bell basis = {title}")
        fig.savefig(f"bell_sweep_{title}.png", dpi=300, bbox_inches="tight")

    def execute_sequence(self, sequence, qubits, nshots):
        platform = self.platform
        results = platform.execute_pulse_sequence(sequence, nshots=nshots)
        frequencies = calculate_frequencies(results[qubits[0]], results[qubits[1]])
        return frequencies

    def execute_circuit(self, circuit, nshots):
        result = circuit(nshots=nshots)
        frequencies = result.frequencies()
        return frequencies

    def execute(
        self,
        qubits,
        bell_basis,
        thetas,
        nshots=1024,
        pulses=False,
        native=True,
        readout_mitigation=None,
        exact=False,
    ):
        """Executes the Bell experiment, with the given bell basis and thetas.
        pulses decides if to execute in the experiment directly in pulses.
        native uses the native interactions but using qibo gates.
        readout_mitigation allows to pass a ReadoutErrorMitigation object.
        exact also computes the exact simulation to compare with noisy results.

        """
        chsh_values_basis = []
        if readout_mitigation:
            mitigated_chsh_values_basis = []
        if exact:
            exact_chsh_values_basis = []

        for basis in bell_basis:
            chsh_values = []
            if readout_mitigation:
                mitigated_chsh_values = []
            if exact:
                exact_chsh_values = []
            for theta in thetas:
                chsh_frequencies = []
                if readout_mitigation:
                    mitigated_chsh_frequencies = []
                if exact:
                    exact_chsh_frequencies = []
                if pulses:
                    chsh_sequences = self.create_chsh_sequences(qubits, theta, basis)
                    for sequence in chsh_sequences:
                        frequencies = self.execute_sequence(sequence, qubits, nshots)
                        chsh_frequencies.append(frequencies)
                else:
                    chsh_circuits = self.create_chsh_circuits(
                        qubits, theta, basis, native
                    )
                    for circuit in chsh_circuits:
                        frequencies = self.execute_circuit(circuit, nshots)
                        chsh_frequencies.append(frequencies)
                if exact:
                    exact_chsh_circuits = self.create_chsh_circuits(
                        qubits, theta, basis, native, rerr=(0.0, 0.0)
                    )
                    for circuit in exact_chsh_circuits:
                        frequencies = self.execute_circuit(circuit, nshots)
                        exact_chsh_frequencies.append(frequencies)

                if readout_mitigation:
                    for frequency in chsh_frequencies:
                        mitigated_frequency = (
                            readout_mitigation.apply_readout_mitigation(frequency)
                        )
                        mitigated_chsh_frequencies.append(mitigated_frequency)

                chsh_bare = self.compute_chsh(chsh_frequencies, basis)
                chsh_values.append(chsh_bare)
                if readout_mitigation:
                    chsh_mitigated = self.compute_chsh(
                        mitigated_chsh_frequencies, basis
                    )
                    mitigated_chsh_values.append(chsh_mitigated)
                if exact:
                    chsh_exact = self.compute_chsh(exact_chsh_frequencies, basis)
                    exact_chsh_values.append(chsh_exact)

            chsh_values_basis.append(chsh_values)
            if readout_mitigation:
                mitigated_chsh_values_basis.append(mitigated_chsh_values)
            if exact:
                exact_chsh_values_basis.append(exact_chsh_values)

        timestr = time.strftime("%Y%m%d-%H%M")

        if readout_mitigation:
            data = {
                "chsh_bare": chsh_values_basis,
                "chsh_mitigated": mitigated_chsh_values_basis,
            }
            with open(f"{timestr}_chsh.json", "w") as file:
                json.dump(data, file)
            if exact:
                for i in range(len(bell_basis)):
                    self.plot(
                        thetas,
                        chsh_values_basis[i],
                        mitigated_chsh_values_basis[i],
                        exact_chsh_values=exact_chsh_values_basis[i],
                        title=bell_basis[i],
                    )
            else:
                for i in range(len(bell_basis)):
                    self.plot(
                        thetas,
                        chsh_values_basis[i],
                        mitigated_chsh_values_basis[i],
                        title=bell_basis[i],
                    )
            # return chsh_values_basis, mitigated_chsh_values_basis
        else:
            data = {"chsh_bare": chsh_values_basis}
            with open(f"{timestr}_chsh.json", "w") as file:
                json.dump(data, file)
            if exact:
                for i in range(len(bell_basis)):
                    self.plot(
                        thetas,
                        chsh_values_basis[i],
                        exact_chsh_values=exact_chsh_values_basis[i],
                        title=bell_basis[i],
                    )
            else:
                for i in range(len(bell_basis)):
                    self.plot(thetas, chsh_values_basis[i], title=bell_basis[i])
            # return chsh_values_basis
