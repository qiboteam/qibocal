import json
import pathlib
from numbers import Number
from typing import Optional, Union

import numpy as np
from qibo import gates
from qibo.models import Circuit

from qibocal.protocols.characterization.utils import significant_digit

SINGLE_QUBIT_CLIFFORDS = {
    # Virtual gates
    0: gates.I,
    1: lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z,
    2: lambda q: gates.U3(q, 0, np.pi / 2, 0),  # gates.RZ(q, np.pi / 2),
    3: lambda q: gates.U3(q, 0, -np.pi / 2, 0),  # gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.U3(q, np.pi, 0, np.pi),  # X,
    5: lambda q: gates.U3(q, np.pi, 0, 0),  # Y,
    # pi/2 rotations
    6: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),  # Rx(pi/2)
    7: lambda q: gates.U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),  # -Rx(pi/2)
    8: lambda q: gates.U3(q, np.pi / 2, 0, 0),  # Ry(pi/2)
    9: lambda q: gates.U3(q, -np.pi / 2, 0, 0),  # -Ry(pi/2)
    # 2pi/3 rotations
    10: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    11: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
    12: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
    13: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
    14: lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    15: lambda q: gates.U3(q, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
    16: lambda q: gates.U3(q, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
    17: lambda q: gates.U3(q, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    18: lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    19: lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    20: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
    21: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
    22: lambda q: gates.U3(q, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    23: lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}

SINGLE_QUBIT_CLIFFORDS_NAMES = {
    # Virtual gates
    "": gates.I,
    # Check the Z
    "minusX,minusY": lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z, gp:exp(iπ)
    "sqrtX,sqrtMinusY,sqrtMinusX": lambda q: gates.U3(
        q, 0, -np.pi / 2, 0
    ),  # La U3 esta bien el nombre no ?  # gates.RZ(q, np.pi / 2),  gp:exp(iπ/4)
    "sqrtX,sqrtY,sqrtMinusX": lambda q: gates.U3(
        q, 0, np.pi / 2, 0
    ),  # gates.U3(q, 0, -np.pi / 2, 0),  # Esta bien gates.RZ(q, -np.pi / 2),  gp:exp(iπ/4)
    # pi rotations
    # 'X': lambda q: gates.U3(q, np.pi, 0, np.pi),  # X,
    # 'Y': lambda q: gates.U3(q, np.pi, 0, 0),  # Y,
    # pi rotations (For the minus exp(iπ) global phase) (Check the phase from qiskit) RX(π)=−iX; RY(π)=−iY
    "minusX": lambda q: gates.U3(q, np.pi, -np.pi, 0),  # X, gp:exp(iπ)
    "minusY": lambda q: gates.U3(q, np.pi, 0, 0),  # Y, gp:exp(iπ)
    # pi/2 rotations (Check the minus) RX(π/2)=−exp(i π/4)SX
    "sqrtX": lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),  # Rx(pi/2) gp:
    "sqrtMinusX": lambda q: gates.U3(
        q, -np.pi / 2, -np.pi / 2, np.pi / 2
    ),  # Rx(-pi/2) gp:
    "sqrtY": lambda q: gates.U3(q, np.pi / 2, 0, 0),  # Ry(pi/2) gp:
    "sqrtMinusY": lambda q: gates.U3(q, -np.pi / 2, 0, 0),  # Ry(-pi/2) gp:
    # 2pi/3 rotations Check the gp
    "sqrtX,sqrtY": lambda q: gates.U3(
        q, np.pi / 2, -np.pi / 2, 0
    ),  # Rx(pi/2)Ry(pi/2) gp:
    "sqrtX,sqrtMinusY": lambda q: gates.U3(
        q, np.pi / 2, -np.pi / 2, np.pi
    ),  # Rx(pi/2)Ry(-pi/2) gp:
    "sqrtMinusX,sqrtY": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, 0
    ),  # Rx(-pi/2)Ry(pi/2) gp:
    "sqrtMinusX,sqrtMinusY": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, -np.pi
    ),  # Rx(-pi/2)Ry(-pi/2) gp:
    "sqrtY,sqrtX": lambda q: gates.U3(
        q, np.pi / 2, 0, np.pi / 2
    ),  # Ry(pi/2)Rx(pi/2) gp:
    "sqrtY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, 0, -np.pi / 2
    ),  # Ry(pi/2)Rx(-pi/2) gp:
    "sqrtMinusY,sqrtX": lambda q: gates.U3(
        q, np.pi / 2, -np.pi, np.pi / 2
    ),  # Ry(-pi/2)Rx(pi/2) gp:
    "sqrtMinusY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, np.pi, -np.pi / 2
    ),  # Ry(-pi/2)Rx(-pi/2) gp:
    # Hadamard-like Check the gp
    "minusX,sqrtY": lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2) gp:
    "minusX,sqrtMinusY": lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2) gp:
    "minusY,sqrtX": lambda q: gates.U3(
        q, np.pi / 2, np.pi / 2, np.pi / 2
    ),  # Y Rx(pi/2) gp:
    "minusY,sqrtMinusX": lambda q: gates.U3(
        q, np.pi / 2, -np.pi / 2, -np.pi / 2
    ),  # Y Rx(-pi/2) gp:
    "sqrtX,sqrtY,sqrtX": lambda q: gates.U3(
        q, np.pi, -np.pi / 4, np.pi / 4
    ),  # Rx(pi/2)Ry(pi/2)Rx(pi/2) gp:
    "sqrtX,sqrtMinusY,sqrtX": lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2) gp:
}


def random_clifford(random_index_gen):
    """Generates random Clifford operator.

    Args:
        qubits (int or list or ndarray): if ``int``, the number of qubits for the Clifford.
            If ``list`` or ``ndarray``, indexes of the qubits for the Clifford to act on.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Default is ``None``.

    Returns:
        (list of :class:`qibo.gates.Gate`): Random Clifford operator(s).
    """

    random_index = int(random_index_gen(SINGLE_QUBIT_CLIFFORDS))
    clifford_gate = SINGLE_QUBIT_CLIFFORDS[random_index](0)

    return clifford_gate, random_index


# TODO: Expand when more entangling gates are calibrated
def find_cliffords(cz_list):
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
    # Separate values containing 1
    values_with_1 = [value for value in clifford if "1" in value]
    values_with_1 = ",".join(values_with_1)

    # Separate values containing 2
    values_with_2 = [value for value in clifford if "2" in value]
    values_with_2 = ",".join(values_with_2)

    # Check if CZ
    value_with_CZ = [value for value in clifford if "CZ" in value]
    value_with_CZ = len(value_with_CZ) == 1

    values_with_1 = values_with_1.replace("1", "")
    values_with_2 = values_with_2.replace("2", "")
    return values_with_1, values_with_2, value_with_CZ


def clifford2gates(clifford):
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


def random_2q_clifford(random_index_gen, two_qubit_cliffords):
    """Generates random two qubit Clifford operator.

    Args:
        qubits (int or list or ndarray): if ``int``, the number of qubits for the Clifford.
            If ``list`` or ``ndarray``, indexes of the qubits for the Clifford to act on.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Default is ``None``.

    Returns:
        (list of :class:`qibo.gates.Gate`): Random Clifford operator(s).
    """

    random_index = int(random_index_gen(two_qubit_cliffords))
    clifford = two_qubit_cliffords[str(random_index)]
    clifford_gate = clifford2gates(clifford)

    return clifford_gate, random_index


def clifford_to_matrix(clifford):
    clifford_gate = clifford2gates(clifford)

    qubits_str = ["q0", "q1"]

    new_circuit = Circuit(2, wire_names=qubits_str)
    for gate in clifford_gate:
        new_circuit.add(gate)

    unitary = new_circuit.unitary()

    return unitary


def generate_inv_dict_cliffords_file(two_qubit_cliffords, output_file):
    """
    Generate an inverse dictionary of clifford matrices and save it to a npz file.

    Parameters:
    two_qubit_cliffords (dict): A dictionary of two-qubit cliffords.
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


def number_to_str(
    value: Number,
    uncertainty: Optional[Union[Number, list, tuple, np.ndarray]] = None,
    precision: Optional[int] = None,
):
    """Converts a number into a string.

    Args:
        value (Number): the number to display
        uncertainty (Number or list or tuple or np.ndarray, optional): number or 2-element
            interval with the low and high uncertainties of ``value``. Defaults to ``None``.
        precision (int, optional): nonnegative number of floating points of the displayed value.
            If ``None``, defaults to the second significant digit of ``uncertainty``
            or ``3`` if ``uncertainty`` is ``None``. Defaults to ``None``.

    Returns:
        str: The number expressed as a string, with the uncertainty if given.
    """

    # If uncertainty is not given, return the value with precision
    if uncertainty is None:
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"

    if isinstance(uncertainty, Number):
        if precision is None:
            precision = (significant_digit(uncertainty) + 1) or 3
        return f"{value:.{precision}f} \u00B1 {uncertainty:.{precision}f}"

    # If any uncertainty is None, return the value with precision
    if any(u is None for u in uncertainty):
        return f"{value:.{precision if precision is not None else 3}f}"

    # If precision is None, get the first significant digit of the uncertainty
    if precision is None:
        precision = max(significant_digit(u) + 1 for u in uncertainty) or 3

    # Check if both uncertainties are equal up to precision
    if np.round(uncertainty[0], precision) == np.round(uncertainty[1], precision):
        return f"{value:.{precision}f} \u00B1 {uncertainty[0]:.{precision}f}"

    return f"{value:.{precision}f} +{uncertainty[1]:.{precision}f} / -{uncertainty[0]:.{precision}f}"


def data_uncertainties(data, method=None, data_median=None, homogeneous=True):
    """Compute the uncertainties of the median (or specified) values.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (float, optional): method of computing the method.
            If it is `None`, computes the standard deviation, otherwise it
            computes the corresponding confidence interval using ``np.percentile``.
            Defaults to ``None``.
        data_median (list or np.ndarray, optional): 1d array for computing the errors from the
            confidence interval. If ``None``, the median values are computed from ``data``.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size
            and returns ``np.ndarray``. Default is ``True``.

    Returns:
        np.ndarray: uncertainties of the data.
    """
    if method is None:
        return np.std(data, axis=1) if homogeneous else [np.std(row) for row in data]

    percentiles = [
        (100 - method) / 2,
        (100 + method) / 2,
    ]
    percentile_interval = np.percentile(data, percentiles, axis=1)

    uncertainties = np.abs(np.vstack([data_median, data_median]) - percentile_interval)

    return uncertainties


def clifford_to_pulses(clifford):
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


def calculate_pulses_clifford(two_qubit_cliffords):
    pulses = 0
    for i, clifford in enumerate(two_qubit_cliffords.values()):
        clifford = two_qubit_cliffords[str(i)]
        pulses += clifford_to_pulses(clifford)

    pulses_per_clifford = pulses / len(two_qubit_cliffords)
    return pulses_per_clifford


def load_inverse_cliffords(file_inv):
    path = pathlib.Path(__file__).parent / file_inv
    clifford_matrices_inv = np.load(path)
    return clifford_matrices_inv


def load_cliffords(file):
    path = pathlib.Path(__file__).parent / "2qubitCliffs.json"
    with open(path) as file:
        two_qubit_cliffords = json.load(file)
    return two_qubit_cliffords
