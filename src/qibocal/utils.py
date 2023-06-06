from typing import Dict, List, Optional, Union

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId


def allocate_qubits(
    platform: Optional[Platform], qubit_ids: List[QubitId]
) -> Union[List[QubitId], Dict[QubitId, Qubit]]:
    """Convert List[QubitId] -> Dict[QubitId, Qubit] for non-trivial platform."""
    qubits = {}
    for q in qubit_ids:
        if not isinstance(q, list) and q in platform.qubits:
            qubits[q] = platform.qubits[q]
        else:
            qubits[tuple(sorted(q))] = platform.pairs[tuple(sorted(q))]
    return qubits
