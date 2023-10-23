from typing import Optional

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPair


def allocate_single_qubits(
    platform: Optional[Platform], qubit_ids: list[QubitId]
) -> dict[QubitId, Qubit]:
    """Construct the map from the chosen ids to the corresponding physical qubits available on the platform."""
    return {q: platform.qubits[q] for q in qubit_ids}


def allocate_qubits_pairs(
    platform: Optional[Platform], qubit_pairs_ids: list[tuple[QubitId, QubitId]]
) -> dict[tuple[QubitId, QubitId], QubitPair]:
    """Construct the map from the chosen id pairs to the corresponding physical qubit pairs available on the platform."""
    return {tuple(qq): platform.pairs[tuple(sorted(qq))] for qq in qubit_pairs_ids}


def allocate_single_qubits_lists(
    platform: Optional[Platform], qubit_lists: list[list[QubitId]]
) -> dict[tuple[QubitId, ...], dict[QubitId, Qubit]]:
    """Construct the map from the chosen id to the corresponding list of physical qubits available on the platform."""
    return {
        tuple(qubits): allocate_single_qubits(platform, qubits)
        for qubits in qubit_lists
    }
