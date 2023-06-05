"""Collection of function to generate qibo circuits."""
from copy import deepcopy
from typing import Callable

from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.models import Circuit


def embed_circuit(circuit: Circuit, nqubits: int, qubits: list) -> Circuit:
    """Embeds `circuit` into a larger circuit of size `nqubits` on the qubits specified by `qubits`.

    Args:
        circuit (Circuit): circuit to be embedded.
        nqubits (int): Number qubits in the large circuit.
        qubits (list): The qubits indices of larger circuit corresponding to qubits of `circuit.

    Returns:
        Circuit: Circuit with nqubits many qubits.
    """

    # get parameters of original circuit
    circuit_init_kwargs = deepcopy(circuit.init_kwargs)
    # Adjust the number of qubits number.
    circuit_init_kwargs["nqubits"] = nqubits
    # Instantiate embedding circuit
    large_circuit = Circuit(**circuit_init_kwargs)
    # Place gates from from original circuit into embedding circuit
    large_circuit.add(circuit.on_qubits(*qubits))
    return large_circuit


def layer_circuit(layer_gen: Callable, depth: int) -> Circuit:
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
    for _ in range(depth):
        # Generate a layer.
        new_layer = layer_gen()
        # Ensure new_layer is a circuit
        if isinstance(new_layer, Gate):
            new_circuit = Circuit(max(new_layer.qubits) + 1)
            new_circuit.add(new_layer)
        elif all(isinstance(gate, Gate) for gate in new_layer):
            new_circuit = Circuit(max(max(gate.qubits) for gate in new_layer) + 1)
            new_circuit.add(new_layer)
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
    return full_circuit


def add_inverse_layer(circuit: Circuit, single_qubit=True):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """

    if circuit.depth > 0:
        if single_qubit:
            # Build an inverse gate for each qubit out of the unitaries using the light cone
            # method and take the daggered version of that.
            for qubit in range(circuit.nqubits):
                circuit_q = circuit.light_cone(qubit)[0]
                circuit.add(gates.Unitary(circuit_q.unitary(), qubit).dagger())
        else:
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that as the last gate.
            circuit.add(
                gates.Unitary(circuit.unitary(), *range(circuit.nqubits)).dagger()
            )


def add_measurement_layer(circuit: Circuit):
    """Adds a measurement layer at the end of the circuit.

    Args:
        circuit (Circuit): Measurement gates added in place to end of this circuit.
    """

    circuit.add(gates.M(*range(circuit.nqubits)))
