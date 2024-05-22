from dataclasses import dataclass

from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    _plot,
)

from .utils import RB2QData, StandardRBResult, fit, twoq_rb_acquisition


@dataclass
class StandardRB2QParameters(StandardRBParameters):
    """Parameters for the standard 2q randomized benchmarking protocol."""

    file: str = "2qubitCliffs.json"
    """File with the cliffords to be used."""
    file_inv: str = "2qubitCliffsInv.npz"
    """File with the cliffords to be used in an inverted dict."""


def _acquisition(
    params: StandardRB2QParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    return twoq_rb_acquisition(params, targets)


def _fit(data: RB2QData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    qubits = data.pairs
    results = fit(qubits, data)

    return results


standard_rb_2q = Routine(_acquisition, _fit, _plot)
