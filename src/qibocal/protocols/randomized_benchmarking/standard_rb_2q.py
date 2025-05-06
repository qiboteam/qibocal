from dataclasses import dataclass

from qibocal.auto.operation import QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    _plot,
)

from ...calibration.calibration import TwoQubitCalibration
from .utils import RB2QData, StandardRBResult, fit, twoq_rb_acquisition

__all__ = ["standard_rb_2q", "StandardRB2QParameters"]


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
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> RB2QData:
    """Data acquisition for two qubit Standard Randomized Benchmarking."""

    return twoq_rb_acquisition(params, platform, targets)


def _fit(data: RB2QData) -> StandardRBResult:
    qubits = data.pairs
    results = fit(qubits, data)

    return results


def _update(
    results: StandardRBResult, platform: CalibrationPlatform, target: QubitPairId
):
    """Write rb fidelity in calibration."""
    if target not in platform.calibration.two_qubits:
        platform.calibration.two_qubits[target] = TwoQubitCalibration()

    platform.calibration.two_qubits[target].rb_fidelity = (
        results.fidelity[target],
        results.fit_uncertainties[target][1] / 2,
    )


standard_rb_2q = Routine(_acquisition, _fit, _plot, _update)
