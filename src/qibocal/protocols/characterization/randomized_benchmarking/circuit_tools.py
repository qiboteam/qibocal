"""Collection of function to generate qibo circuits."""

from typing import Callable

import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.models import Circuit


def layer_circuit(layer_gen: Callable, depth: int, qubit, seed) -> Circuit:
    """Creates a circuit of `depth` layers from a generator `layer_gen` yielding `Circuit` or `Gate`.

    Args:
        layer_gen (Callable): Should return gates or a full circuit specifying a layer.
        depth (int): Number of layers.

    Returns:
        Circuit: with `depth` many layers.
    """

    if not isinstance(depth, int) or depth <= 0:
        raise_error(ValueError, "Depth must be type int and positive.")
    full_circuit = None
    # Build each layer, there will be depth many in the final circuit.
    qubits_str = [str(qubit)]
    random_indexes = []
    for _ in range(depth):
        # Generate a layer.
        new_layer, random_index = layer_gen(1, seed)  # TODO: find better implementation
        random_indexes.append(random_index[0])
        # Ensure new_layer is a circuit
        if isinstance(new_layer, Gate):
            new_circuit = Circuit(1, wire_names=qubits_str)
            new_circuit.add(new_layer)
        elif all(isinstance(gate, Gate) for gate in new_layer):
            new_circuit = Circuit(1, wire_names=qubits_str)

            new_circuit.add(new_layer)
        elif isinstance(new_layer, Circuit):
            new_circuit = new_layer
        else:
            raise_error(
                TypeError,
                f"layer_gen must return type Circuit or Gate, but it is type {type(new_layer)}.",
            )
        if full_circuit is None:  # instantiate in first loop
            full_circuit = Circuit(new_circuit.nqubits, wire_names=qubits_str)
        full_circuit = full_circuit + new_circuit

    with open("random_indexes.csv", "a") as file:
        np.savetxt(file, np.asarray([random_indexes]), delimiter=",", fmt="%d")

    return full_circuit


def add_inverse_layer(circuit: Circuit, single_qubit=True):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """

    if circuit.depth > 0:
        circuit.add(gates.Unitary(circuit.unitary(), *range(circuit.nqubits)).dagger())


def add_measurement_layer(circuit: Circuit):
    """Adds a measurement layer at the end of the circuit.

    Args:
        circuit (Circuit): Measurement gates added in place to end of this circuit.
    """

    circuit.add(gates.M(*range(circuit.nqubits)))
