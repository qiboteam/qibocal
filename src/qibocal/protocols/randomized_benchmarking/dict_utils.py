import json
import pathlib

import numpy as np
from qibo import gates
from qibo.models import Circuit

SINGLE_QUBIT_CLIFFORDS_NAMES = {
    # Virtual gates
    "": gates.I,
    "minusX,minusY": lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z
    "sqrtX,sqrtMinusY,sqrtMinusX": lambda q: gates.U3(
        q, 0, -np.pi / 2, 0
    ),  # gates.RZ(q, np.pi / 2)
    "sqrtX,sqrtY,sqrtMinusX": lambda q: gates.U3(
        q, 0, np.pi / 2, 0
    ),  # gates.RZ(q, -np.pi / 2)
    # pi rotations
    "minusX": lambda q: gates.U3(q, np.pi, -np.pi, 0),  # X
    "minusY": lambda q: gates.U3(q, np.pi, 0, 0),  # Y
    # pi/2 rotations
    "sqrtX": lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),  # Rx(pi/2)
    "sqrtMinusX": lambda q: gates.U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),  # Rx(-pi/2)
    "sqrtY": lambda q: gates.U3(q, np.pi / 2, 0, 0),  # Ry(pi/2)
    "sqrtMinusY": lambda q: gates.U3(q, -np.pi / 2, 0, 0),  # Ry(-pi/2)
    # 2pi/3 rotations
    "sqrtX,sqrtY": lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    "sqrtX,sqrtMinusY": lambda q: gates.U3(
        q, np.pi / 2, -np.pi / 2, np.pi
    ),  # Rx(pi/2)Ry(-pi/2)
    "sqrtMinusX,sqrtY": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, 0
    ),  # Rx(-pi/2)Ry(pi/2)
    "sqrtMinusX,sqrtMinusY": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, -np.pi
    ),  # Rx(-pi/2)Ry(-pi/2)
    "sqrtY,sqrtX": lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    "sqrtY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, 0, -np.pi / 2
    ),  # Ry(pi/2)Rx(-pi/2)
    "sqrtMinusY,sqrtX": lambda q: gates.U3(
        q, np.pi / 2, -np.pi, np.pi / 2
    ),  # Ry(-pi/2)Rx(pi/2)
    "sqrtMinusY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, np.pi, -np.pi / 2
    ),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    "minusX,sqrtY": lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    "minusX,sqrtMinusY": lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    "minusY,sqrtX": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, np.pi / 2
    ),  # Y Rx(pi/2)
    "minusY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, -np.pi / 2, -np.pi / 2
    ),  # Y Rx(-pi/2)
    "sqrtX,sqrtY,sqrtX": lambda q: gates.U3(
        q, np.pi, -np.pi / 4, np.pi / 4
    ),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    "sqrtX,sqrtMinusY,sqrtX": lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}


# TODO: Expand when more entangling gates are calibrated
def find_cliffords(cz_list):
    """Splits a clifford (list of gates) into sublists based on the occurrence of the "CZ" gate."""
    clifford_list = []
    clifford = []
    for gate in cz_list:
        if gate == "CZ":
            clifford.append(gate)
            clifford_list.append(clifford)
            clifford = []
            continue
        clifford.append(gate)
    clifford_list.append(clifford)
    return clifford_list


def separator(clifford):
    """
    Separates values in the given clifford sublist based on certain conditions.

    Returns:
        tuple: A tuple containing three elements:
            - values_with_1 (str): A comma-separated string of values containing '1'.
            - values_with_2 (str): A comma-separated string of values containing '2'.
            - value_with_CZ (bool): True if 'CZ' is present in the clifford list, False otherwise.
    """

    # Separate values containing 1
    values_with_1 = [value for value in clifford if "1" in value]
    values_with_1 = ",".join(values_with_1)

    # Separate values containing 2
    values_with_2 = [value for value in clifford if "2" in value]
    values_with_2 = ",".join(values_with_2)

    # Check if CZ
    value_with_CZ = [value for value in clifford if "CZ" in value]
    value_with_CZ = len(value_with_CZ) == 1  # FIXME: What is this ?

    values_with_1 = values_with_1.replace("1", "")
    values_with_2 = values_with_2.replace("2", "")
    return values_with_1, values_with_2, value_with_CZ


def clifford2gates(clifford):
    """
    Converts a Clifford string into a list of gates.

    Args:
        clifford (str): A comma-separated string representing a sequence of gates that represent a Clifford gate.
    """
    gate_list = clifford.split(",")

    clifford_list = find_cliffords(gate_list)

    clifford_gate = []
    for clifford in clifford_list:
        values_with_1, values_with_2, value_with_CZ = separator(clifford)
        clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_1](0))
        clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_2](1))
        if value_with_CZ:
            clifford_gate.append(gates.CZ(0, 1))

    return clifford_gate


def clifford_to_matrix(clifford):
    """
    Converts a Clifford gate as a string to its corresponding unitary matrix representation.
    """
    clifford_gate = clifford2gates(clifford)

    qubits_str = ["q0", "q1"]

    new_circuit = Circuit(2, wire_names=qubits_str)
    for gate in clifford_gate:
        new_circuit.add(gate)

    unitary = new_circuit.unitary()

    return unitary


def generate_inv_dict_cliffords_file(two_qubit_cliffords, output_file=None):
    """
    Generate an inverse dictionary of Clifford matrices and save it to a npz file.

    Parameters:
    two_qubit_cliffords (dict): A dictionary of two-qubit Cliffords.
    output_file (str): The path to the output npz file.
    """
    clifford_matrices = {}
    for i, clifford in enumerate(two_qubit_cliffords.values()):
        clifford = two_qubit_cliffords[str(i)]

        unitary = clifford_to_matrix(clifford)
        unitary = unitary.round(3)
        unitary += 0.0 + 0.0j

        clifford_matrices[i] = unitary

    clifford_matrices_inv_np = {}
    # Convert the arrays to strings and store them as keys in the new dictionary
    for key, value in clifford_matrices.items():
        key_str = np.array2string(value, separator=",")
        clifford_matrices_inv_np[key_str] = key

    if output_file is not None:
        np.savez(output_file, **clifford_matrices_inv_np)

    return clifford_matrices_inv_np


def clifford_to_pulses(clifford):
    """
    From a Clifford gate sequence into the number of pulses required to implement it.

    Args:
        clifford (str): A comma-separated string representing the Clifford gate sequence.

    Returns:
        int: The number of pulses required to implement the given Clifford gate sequence.
    """
    gate_list = clifford.split(",")

    clifford_list = find_cliffords(gate_list)

    pulses = 0
    for clifford in clifford_list:
        values_with_1, values_with_2, value_with_CZ = separator(clifford)

        if SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_1](0).name != "id":
            pulses += 2  # This assumes a U3 transpiled into 2 pulses
        if SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_2](1).name != "id":
            pulses += 2  # This assumes a U3 transpiled into 2 pulses
        if value_with_CZ:
            pulses += 1  # This assumes a CZ without parking so 1 pulse

    return pulses


def calculate_pulses_clifford(cliffords):
    """
    Calculate the average number of pulses per Clifford operation.

    Parameters:
    - cliffords (dict): A dictionary of Clifford operations.

    Returns:
    - pulses_per_clifford (float): The average number of pulses per Clifford operation.
    """
    pulses = 0
    for i, clifford in enumerate(cliffords.values()):
        clifford = cliffords[str(i)]
        pulses += clifford_to_pulses(clifford)

    pulses_per_clifford = pulses / len(cliffords)
    return pulses_per_clifford


def load_inverse_cliffords(file_inv):
    path = pathlib.Path(__file__).parent / file_inv
    clifford_matrices_inv = np.load(path)
    return clifford_matrices_inv


def load_cliffords(file_cliffords):
    path = pathlib.Path(__file__).parent / file_cliffords
    with open(path) as file:
        two_qubit_cliffords = json.load(file)
    return two_qubit_cliffords
