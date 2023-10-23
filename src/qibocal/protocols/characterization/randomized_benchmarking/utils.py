from numbers import Number
from typing import Callable, Optional, Union

import numpy as np
from pandas import DataFrame
from qibo import gates

from qibocal.config import raise_error
from qibocal.protocols.characterization.utils import significant_digit

SINGLE_QUBIT_CLIFFORDS = {
    # Virtual gates
    0: gates.I,
    1: gates.Z,
    2: lambda q: gates.RZ(q, np.pi / 2),
    3: lambda q: gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: gates.X,  # U3(q, np.pi, 0, np.pi),
    5: gates.Y,  # U3(q, np.pi, 0, 0),
    # pi/2 rotations
    6: lambda q: gates.RX(q, np.pi / 2),  # U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),
    7: lambda q: gates.RX(q, -np.pi / 2),  # U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),
    8: lambda q: gates.RY(q, np.pi / 2),  # U3(q, np.pi / 2, 0, 0),
    9: lambda q: gates.RY(q, -np.pi / 2),  # U3(q, -np.pi / 2, 0, 0),
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


def random_clifford(qubits, seed=None):
    """Generates random Clifford operator(s).

    Args:
        qubits (int or list or ndarray): if ``int``, the number of qubits for the Clifford.
            If ``list`` or ``ndarray``, indexes of the qubits for the Clifford to act on.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Default is ``None``.

    Returns:
        (list of :class:`qibo.gates.Gate`): Random Clifford operator(s).
    """

    if (
        not isinstance(qubits, int)
        and not isinstance(qubits, list)
        and not isinstance(qubits, np.ndarray)
    ):
        raise_error(
            TypeError,
            f"qubits must be either type int, list or ndarray, but it is type {type(qubits)}.",
        )
    if isinstance(qubits, int) and qubits <= 0:
        raise_error(ValueError, "qubits must be a positive integer.")

    if isinstance(qubits, (list, np.ndarray)) and any(q < 0 for q in qubits):
        raise_error(ValueError, "qubit indexes must be non-negative integers.")

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    if isinstance(qubits, int):
        qubits = list(range(qubits))

    random_indexes = local_state.integers(0, len(SINGLE_QUBIT_CLIFFORDS), len(qubits))
    clifford_gates = [
        SINGLE_QUBIT_CLIFFORDS[p](q) for p, q in zip(random_indexes, qubits)
    ]

    return clifford_gates


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


def extract_from_data(
    data: Union[list[dict], DataFrame],
    output_key: str,
    groupby_key: str = "",
    agg_type: Union[str, Callable] = "",
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return wanted values from list of dictionaries via a dataframe and its properties.

    If ``groupby_key`` given, aggregate the dataframe, extract the data by which the frame was
    grouped, what was calculated given the ``agg_type`` parameter. Two arrays are returned then,
    the group values and the grouped (aggregated) data. If no ``agg_type`` given use a linear
    function. If ``groupby_key`` not given, only return the extracted data from given key.

    Args:
        output_key (str): Key name of the wanted output.
        groupby_key (str): If given, group with that key name.
        agg_type (str): If given, calcuted aggregation function on groups.

    Returns:
        Either one or two np.ndarrays. If no grouping wanted, just the data. If grouping
        wanted, the values after which where grouped and the grouped data.
    """
    if isinstance(data, list):
        data = DataFrame(data)

    # Check what parameters where given.
    if not groupby_key and not agg_type:
        # No grouping and no aggreagtion is wanted. Just return the wanted output key.
        return np.array(data[output_key].to_numpy())
    if not groupby_key and agg_type:
        # No grouping wanted, just an aggregational task on all the data.
        return data[output_key].apply(agg_type)
    if groupby_key and not agg_type:
        df = data.get([output_key, groupby_key])
        # Sort by the output key for making reshaping consistent.
        df.sort_values(by=output_key)
        # Grouping is wanted but no aggregation, use a linear function.
        grouped_df = df.groupby(groupby_key, group_keys=True).apply(lambda x: x)
        return grouped_df[groupby_key].to_numpy(), grouped_df[output_key].to_numpy()
    df = data.get([output_key, groupby_key])
    grouped_df = df.groupby(groupby_key, group_keys=True).agg(agg_type)
    return grouped_df.index.to_numpy(), grouped_df[output_key].values.tolist()
