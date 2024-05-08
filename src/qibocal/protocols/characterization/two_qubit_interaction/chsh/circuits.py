"""Auxiliary functions to run CHSH using circuits."""

import numpy as np
from qibo import gates
from qibo.models import Circuit

from .utils import READOUT_BASIS


def create_bell_circuit(nqubits, qubits, theta=np.pi / 4, bell_state=0):
    """Creates the circuit to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    Native defaults to only using GPI2 and GPI gates.
    """
    p = [0, 0]
    c = Circuit(nqubits)
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


def create_bell_circuit_native(nqubits, qubits, theta=np.pi / 4, bell_state=0):
    """Creates the circuit to generate the bell states and with a theta-measurement
    bell_state chooses the initial bell state for the test:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    Native defaults to only using GPI2 and GPI gates.
    """

    c = Circuit(nqubits)
    p = [0, 0]
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
        c.add(gates.GPI2(qubits[0], p[0]))
        c.add(gates.GPI2(qubits[0], p[0]))
    elif bell_state == 3:
        p[0] += np.pi
        c.add(gates.GPI2(qubits[0], p[0]))
        c.add(gates.GPI2(qubits[0], p[0]))

    c.add(gates.GPI2(qubits[0], p[0]))
    p[0] += theta
    c.add(gates.GPI2(qubits[0], p[0] + np.pi))

    return c, p


def create_chsh_circuits(
    platform,
    qubits,
    theta=np.pi / 4,
    bell_state=0,
    native=True,
    readout_basis=READOUT_BASIS,
):
    """Creates the circuits needed for the 4 measurement settings for chsh.
    Native defaults to only using GPI2 and GPI gates.
    rerr adds a readout bitflip error to the simulation.
    """
    create_bell = create_bell_circuit_native if native else create_bell_circuit
    chsh_circuits = {}
    nqubits = platform.nqubits if platform else max(qubits) + 1
    for basis in readout_basis:
        c, p = create_bell(nqubits, qubits, theta, bell_state)
        for i, base in enumerate(basis):
            if base == "X":
                if native:
                    c.add(gates.GPI2(qubits[i], p[i] + np.pi / 2))
                else:
                    c.add(gates.H(qubits[i]))
        for qubit in qubits:
            c.add(gates.M(qubit))
        chsh_circuits[basis] = c
    return chsh_circuits
