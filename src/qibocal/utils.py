from typing import Optional

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPair


def cast_to_int(a):
    try:
        return int(a)
    except:
        # remove double quotes
        return a[1:-1]


def conversion(name: str):
    convert_to_list = name.strip("()").split(",")
    return (
        tuple(cast_to_int(i) for i in convert_to_list)
        if len(convert_to_list) > 1
        else cast_to_int(name)
    )


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
