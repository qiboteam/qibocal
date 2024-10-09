import numpy as np
from qibo import Circuit, gates


def create_mermin_circuit(qubits, native=True, theta=None):
    """Creates the circuit to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    Native defaults to only using GPI2 and GPI gates.
    """
    if not theta:
        theta = ((n - 1) * 0.25 * np.pi) % (2 * np.pi)
    # TODO: implement condition better
    # if qubits[1] != 2:
    #     raise ValueError('The center qubit should be in qubits[1]!')
    c = Circuit(len(qubits))
    p = [0, 0, 0]
    if native:
        # TODO: not hardcode connections
        # Centermost qubit is qubits[0]
        for i in range(len(qubits)):
            c.add(gates.GPI2(qubits[i], np.pi / 2))
        for i in range(1, len(qubits)):
            c.add(gates.CZ(qubits[0], qubits[i]))
        for i in range(1, len(qubits)):
            c.add(gates.GPI2(qubits[i], -np.pi / 2))
        p[0] -= theta

    else:
        # TODO: not hardcode connections
        # Centermost qubit is qubits[0]
        for i in range(len(qubits)):
            c.add(gates.H(qubits[i]))
        for i in range(1, len(qubits)):
            c.add(gates.CZ(qubits[0], qubits[i]))
        for i in range(1, len(qubits)):
            c.add(gates.H(qubits[i]))
        c.add(gates.U1(0, theta))
    return c, p


def create_mermin_circuits(qubits, readout_basis, native=True, rerr=None):
    """Creates the circuits needed for the 4 measurement settings for chsh.
    Native defaults to only using GPI2 and GPI gates.
    rerr adds a readout bitflip error to the simulation.
    """

    mermin_circuits = {}

    for basis in readout_basis:
        c, p = create_mermin_circuit(qubits, native)
        for i, base in enumerate(basis):
            if base == "X":
                if native:
                    c.add(gates.GPI2(qubits[i], p[i] + np.pi / 2))
                else:
                    c.add(gates.H(qubits[i]))
            elif base == "Y":
                if native:
                    c.add(gates.GPI2(qubits[i], p[i]))
                else:
                    c.add(gates.SDG(qubits[i]))
                    c.add(gates.H(qubits[i]))

        for qubit in qubits:
            c.add(gates.M(qubit))
            # c.add(gates.M(qubit, p0=rerr[0], p1=rerr[1]))
        mermin_circuits[basis] = c

    return mermin_circuits
