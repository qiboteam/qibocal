from qibolab.qubits import Qubit, QubitId


def order_pair(pair: list[QubitId, QubitId], qubits: dict[QubitId, Qubit]) -> tuple:
    """Order a pair of qubits by drive frequency."""
    if qubits[pair[0]].drive_frequency > qubits[pair[1]].drive_frequency:
        return pair[1], pair[0]
    return pair[0], pair[1]
