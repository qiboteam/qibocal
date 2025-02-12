from dataclasses import dataclass

from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    _plot,
)

from .utils import RB2QData, StandardRBResult, fit, twoq_rb_acquisition

FILE_CLIFFORDS = "2qubitCliffs.json"
FILE_INV = "2qubitCliffsInv.npz"


@dataclass
class StandardRB2QParameters(StandardRBParameters):
    """Parameters for the standard 2q randomized benchmarking protocol."""

    file: str = FILE_CLIFFORDS
    """File with the cliffords to be used."""
    file_inv: str = FILE_INV
    """File with the cliffords to be used in an inverted dict."""


def _acquisition(
    params: StandardRB2QParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QData:
    """Data acquisition for two qubit Standard Randomized Benchmarking."""

    return twoq_rb_acquisition(params, platform, targets)


def _fit(data: RB2QData) -> StandardRBResult:
    qubits = data.pairs
    results = fit(qubits, data)

    return results


standard_rb_2q = Routine(_acquisition, _fit, _plot)
