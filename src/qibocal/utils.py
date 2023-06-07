from typing import Optional, Union

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPair


def allocate_qubits(
    platform: Optional[Platform], qubit_ids: list[QubitId]
) -> Union[dict[QubitId, Qubit], dict[tuple, QubitPair]]:
    """Convert list[QubitId] -> Union[dict[QubitId, Qubit], dict[tuple, QubitPair]] for non-trivial platform."""
    qubits = {}
    for q in qubit_ids:
        if not isinstance(q, list) and q in platform.qubits:
            qubits[q] = platform.qubits[q]
        else:
            qubits[tuple(sorted(q))] = platform.pairs[tuple(sorted(q))]
    return qubits
