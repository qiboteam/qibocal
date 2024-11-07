"""Collection of function to generate qibo circuits."""

import pathlib
from typing import Callable

import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.models import Circuit

from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    SINGLE_QUBIT_CLIFFORDS_NAMES,
    find_cliffords,
    generate_inv_dict_cliffords_file,
    separator,
)

GLOBAL_PHASES = [
    1 + 0j,
    -1 + 0j,
    0 + 1j,
    0 - 1j,
    0.707 + 0.707j,
    -0.707 + 0.707j,
    0.707 - 0.707j,
    -0.707 - 0.707j,
]


def layer_circuit(rb_gen: Callable, depth: int, qubit) -> tuple[Circuit, dict]:
    """Creates a circuit of `depth` layers from a generator `layer_gen` yielding `Circuit` or `Gate`
    and a dictionary with random indexes used to select the clifford gates.

    Args:
        layer_gen (Callable): Should return gates or a full circuit specifying a layer.
        depth (int): Number of layers.

    Returns:
        Circuit: with `depth` many layers.
    """

    full_circuit = None
    random_indexes = []
    # Build each layer, there will be depth many in the final circuit.
    qubits_str = [str(qubit)]

    for _ in range(depth):
        # Generate a layer.
        new_layer, random_index = rb_gen.layer_gen()
        # Ensure new_layer is a circuit
        if isinstance(new_layer, Gate):
            new_circuit = Circuit(1)
            new_circuit.add(new_layer)
            random_indexes.append(random_index)

        # We are only using this for the RB we have right now
        elif all(isinstance(gate, Gate) for gate in new_layer):
            new_circuit = Circuit(1, wire_names=qubits_str)
            new_circuit.add(new_layer)
            random_indexes.append(random_index)

        elif isinstance(new_layer, Circuit):
            new_circuit = new_layer
        else:
            raise_error(
                TypeError,
                f"layer_gen must return type Circuit or Gate, but it is type {type(new_layer)}.",
            )
        if full_circuit is None:  # instantiate in first loop
            full_circuit = Circuit(new_circuit.nqubits)
        full_circuit = full_circuit + new_circuit
    return full_circuit, random_indexes


def layer_2q_circuit(rb_gen: Callable, depth: int, qubits) -> tuple[Circuit, dict]:
    """Creates a circuit of `depth` layers from a generator `layer_gen` yielding `Circuit` or `Gate`
    and a dictionary with random indexes used to select the clifford gates.

    Args:
        layer_gen (Callable): Should return gates or a full circuit specifying a layer.
        depth (int): Number of layers.

    Returns:
        Circuit: with `depth` many layers.
    """

    full_circuit = None
    random_indexes = []
    # Build each layer, there will be depth many in the final circuit.
    for _ in range(depth):
        # Generate a layer.
        new_layer, random_index = rb_gen.layer_gen()
        new_circuit = Circuit(2)
        for gate in new_layer:
            new_circuit.add(gate)
        random_indexes.append(random_index)

        if full_circuit is None:  # instantiate in first loop
            full_circuit = Circuit(new_circuit.nqubits)
        full_circuit = full_circuit + new_circuit
    return full_circuit, random_indexes


def add_inverse_layer(circuit: Circuit, single_qubit=True):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """

    if circuit.depth > 0:
        circuit.add(gates.Unitary(circuit.unitary(), *range(circuit.nqubits)).dagger())


def add_inverse_2q_layer(circuit: Circuit, two_qubit_cliffords, file_inv):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """

    path = pathlib.Path(__file__).parent / file_inv
    if file_inv is None and not path.is_file():
        clifford_matrices_inv = generate_inv_dict_cliffords_file(
            two_qubit_cliffords, file_inv
        )
    else:
        clifford_matrices_inv = np.load(path)

    if circuit.depth > 0:
        clifford = circuit.unitary()

        cliffords = [clifford * global_phase for global_phase in GLOBAL_PHASES]
        cliffords_inv = [np.linalg.inv(clifford).round(3) for clifford in cliffords]

        for clifford_inv in cliffords_inv:
            clifford_inv += 0.0 + 0.0j
            clifford_inv_str = np.array2string(clifford_inv, separator=",")
            if clifford_inv_str in clifford_matrices_inv.files:
                index_inv = clifford_matrices_inv[clifford_inv_str]

        clifford = two_qubit_cliffords[str(index_inv)]

        gate_list = clifford.split(",")

        clifford_list = find_cliffords(gate_list)

        clifford_gate = []
        for clifford in clifford_list:
            values_with_1, values_with_2, value_with_CZ = separator(clifford)
            clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_1](0))
            clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_2](1))
            if value_with_CZ:
                clifford_gate.append(gates.CZ(0, 1))

        for gate in clifford_gate:
            circuit.add(gate)


def add_measurement_layer(circuit: Circuit):
    """Adds a measurement layer at the end of the circuit.

    Args:
        circuit (Circuit): Measurement gates added in place to end of this circuit.
    """

    circuit.add(gates.M(*range(circuit.nqubits)))
