from typing import Dict, List, Union

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId


def allocate_qubits(
    platform: Platform, qubit_ids: List[QubitId]
) -> Union[List[QubitId], Dict[QubitId, Qubit]]:
    """Convert List[QubitId] -> Dict[QubitId, Qubit] for non-trivial platform."""
    if platform is not None:
        return {q: platform.qubits[q] for q in qubit_ids if q in platform.qubits}
    return qubit_ids
