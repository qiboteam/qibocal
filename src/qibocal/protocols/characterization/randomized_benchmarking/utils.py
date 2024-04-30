import json
from numbers import Number
from typing import Optional, Union

import numpy as np
from qibo import gates

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
    "sqrtX,sqrtMinusY,sqrtMinusX": lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z,
    "sqrtX,sqrtY,sqrtMinusX": lambda q: gates.U3(
        q, 0, np.pi / 2, 0
    ),  # gates.RZ(q, np.pi / 2),
    "minusX,minusY": lambda q: gates.U3(
        q, 0, -np.pi / 2, 0
    ),  # gates.RZ(q, -np.pi / 2),
    # pi rotations
    "minusX": lambda q: gates.U3(q, np.pi, 0, np.pi),  # X,
    "minusY": lambda q: gates.U3(q, np.pi, 0, 0),  # Y,
    # pi/2 rotations (Check the minus)
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

with open("2qubitCliffs.json") as file:
    TWO_QUBIT_CLIFFORDS = json.load(file)


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


def random_2q_clifford(random_index_gen):
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

    random_index = int(random_index_gen(TWO_QUBIT_CLIFFORDS))
    clifford = TWO_QUBIT_CLIFFORDS[str(random_index)]

    gate_list = clifford.split(",")

    clifford_list = find_cliffords(gate_list)

    clifford_gate = []
    for clifford in clifford_list:

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

        clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_1](0))
        clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_2](1))
        if value_with_CZ:
            clifford_gate.append(gates.CZ(0, 1))

    return clifford_gate, random_index


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
