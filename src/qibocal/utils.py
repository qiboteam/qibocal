from typing import Optional

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPair


def allocate_single_qubits(
    platform: Optional[Platform], qubit_ids: list[QubitId]
) -> dict[QubitId, Qubit]:
    """Convert list[QubitId] -> dict[QubitId, Qubit] for non-trivial platform."""
    return {platform.qubits[q].name: platform.qubits[q] for q in qubit_ids}


def allocate_qubits_pairs(
    platform: Optional[Platform], qubit_pairs_ids: list[tuple[QubitId, QubitId]]
) -> dict[tuple[QubitId, QubitId], QubitPair]:
    """Convert  list[tuple[QubitId,QubitId]] -> dict[tuple[QubitId,QubitId], QubitPair] for non-trivial platform."""
    return {tuple(qq): platform.pairs[tuple(sorted(qq))] for qq in qubit_pairs_ids}
