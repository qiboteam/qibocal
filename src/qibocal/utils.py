from typing import Dict, List, Optional

from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId


def cast_str_to_int(key):
    if key.isnumeric():
        return int(key)
    return key


def my_eval(key):
    if key.startswith("("):
        test = key[key.find("(") + 1 : key.find(")")].replace(" ", "").split(",")
        return tuple([cast_str_to_int(i) for i in test])

    return cast_str_to_int(key)


def allocate_qubits(
    platform: Optional[Platform], qubit_ids: List[QubitId]
) -> Dict[QubitId, Qubit]:
    """Convert List[QubitId] -> Dict[QubitId, Qubit] for non-trivial platform."""
    return {q: platform.qubits[q] for q in qubit_ids if q in platform.qubits}
