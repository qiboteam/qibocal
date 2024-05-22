from dataclasses import dataclass

import numpy as np
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb import _plot
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRB2QParameters,
)

from .utils import RB2QInterData, StandardRBResult, fit, twoq_rb_acquisition


@dataclass
class StandardRB2QInterParameters(StandardRB2QParameters):
    """Parameters for the standard 2q randomized benchmarking protocol."""

    interleave: str = "CZ"
    """Gate to interleave"""


@dataclass
class StandardRB2QInterResult(StandardRBResult):
    """Standard RB outputs."""

    fidelity_cz: dict[QubitPairId, list] = None
    """The overall fidelity for the CZ gate and its uncertainty."""


def _acquisition(
    params: StandardRB2QInterParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QInterData:
    """Data acquisition for two qubit Standard Randomized Benchmarking."""

    data = twoq_rb_acquisition(params, targets, interleave=params.interleave)

    fidelity = {}
    for target in targets:
        fidelity[target] = platform.pairs[target].gate_fidelity
    data.fidelity = fidelity

    return data


def _fit(data: RB2QInterData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    qubits = data.pairs
    results = fit(qubits, data)

    # FIXME: I can only get the data.fidelity if there is an acquisition step
    if data.fidelity is not None:
        fidelity_cz = {}
        for qubit in qubits:
            fid_cz = results.fidelity[qubit] / data.fidelity[qubit][0]
            uncertainty_cz = np.sqrt(
                1
                / data.fidelity[qubit][0] ** 2
                * results.fit_uncertainties[qubit][1] ** 2
                + (results.fidelity[qubit] / data.fidelity[qubit][0] ** 2) ** 2
                * data.fidelity[qubit][1] ** 2
            )
            fidelity_cz[qubit] = [fid_cz, uncertainty_cz]

        results = StandardRB2QInterResult(
            results.fidelity,
            results.pulse_fidelity,
            results.fit_parameters,
            results.fit_uncertainties,
            results.error_bars,
            fidelity_cz,
        )

    return results


standard_rb_2q_inter = Routine(_acquisition, _fit, _plot)
