from collections import namedtuple

from qibolab.qubits import Qubit, QubitId

OrderedPair = namedtuple("Pair", "low_freq high_freq")
"""Pair object to discriminate high and low freq qubit."""


def order_pair(
    pair: list[QubitId, QubitId], qubits: dict[QubitId, Qubit]
) -> OrderedPair:
    """Order a pair of qubits by drive frequency."""
    if qubits[pair[0]].drive_frequency > qubits[pair[1]].drive_frequency:
        return OrderedPair(pair[1], pair[0])
    return OrderedPair(pair[0], pair[1])
