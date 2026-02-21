from typing import cast

from qibolab import Platform
from qibolab._core.native import SingleQubitNatives
from qibolab._core.parameters import TwoQubitContainer

from .calibration import (
    Calibration,
    QubitCalibration,
    QubitId,
    QubitPairId,
    TwoQubitCalibration,
)

__all__ = ["initialize_calibration"]


def _single_qubits(
    natives: dict[QubitId, SingleQubitNatives],
) -> dict[QubitId, QubitCalibration]:
    return {q: QubitCalibration() for q in natives}


def _two_qubits(
    natives: TwoQubitContainer,
) -> dict[QubitPairId, TwoQubitCalibration]:
    return {cast(QubitPairId, pair): TwoQubitCalibration() for pair in natives}


def initialize_calibration(platform: Platform) -> Calibration:
    single_qubits = _single_qubits(platform.natives.single_qubit)
    two_qubits = _two_qubits(platform.natives.two_qubit)

    return Calibration(single_qubits=single_qubits, two_qubits=two_qubits)
