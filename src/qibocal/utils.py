from typing import Optional

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId


def allocate_qubits(
    platform: Optional[Platform], qubit_ids: list[QubitId]
) -> dict[QubitId, Qubit]:
    """Convert list[QubitId] -> dict[QubitId, Qubit] for non-trivial platform."""
    return {q: platform.qubits[q] for q in qubit_ids if q in platform.qubits}
