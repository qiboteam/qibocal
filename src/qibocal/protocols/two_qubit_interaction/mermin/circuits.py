import numpy as np
from qibo import Circuit, gates


def create_mermin_circuit(qubits, native=True):
    """Creates the circuit to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    Native defaults to only using GPI2 and GPI gates.
    """
    # TODO: implement condition better
    # if qubits[1] != 2:
    #     raise ValueError('The center qubit should be in qubits[1]!')
    c = Circuit(len(qubits))
    p = [0, 0, 0]
    if native:
        c.add(gates.GPI2(qubits[1], np.pi / 2))
        c.add(gates.GPI2(qubits[0], np.pi / 2))
        c.add(gates.CZ(qubits[1], qubits[0]))
        c.add(gates.GPI2(qubits[0], -np.pi / 2))
        c.add(gates.GPI2(qubits[2], np.pi / 2))
        c.add(gates.CZ(qubits[1], qubits[2]))
        c.add(gates.GPI2(qubits[2], -np.pi / 2))
        p[0] -= np.pi / 2

    else:
        c.add(gates.H(qubits[1]))
        c.add(gates.H(qubits[0]))
        c.add(gates.CZ(qubits[1], qubits[0]))
        c.add(gates.H(qubits[0]))
        c.add(gates.H(qubits[2]))
        c.add(gates.CZ(qubits[1], qubits[2]))
        c.add(gates.H(qubits[2]))
        c.add(gates.S(0))
    return c, p


def create_mermin_circuits(qubits, readout_basis, native=True, rerr=None):
    """Creates the circuits needed for the 4 measurement settings for chsh.
    Native defaults to only using GPI2 and GPI gates.
    rerr adds a readout bitflip error to the simulation.
    """

    mermin_circuits = []

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
        mermin_circuits.append(c)

    return mermin_circuits
