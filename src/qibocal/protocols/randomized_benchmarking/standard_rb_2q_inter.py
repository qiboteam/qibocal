from dataclasses import dataclass, fields

import numpy as np

from qibocal.auto.operation import QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.standard_rb import _plot
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRB2QParameters,
)

from .utils import RB2QInterData, StandardRBResult, fit, twoq_rb_acquisition

__all__ = ["standard_rb_2q_inter"]


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

    def __contains__(self, value: QubitPairId):
        return all(
            value in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
            and field.name != "fidelity_cz"
        )


def _acquisition(
    params: StandardRB2QInterParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> RB2QInterData:
    """Data acquisition for two qubit Interleaved Randomized Benchmarking."""
    data = twoq_rb_acquisition(params, platform, targets, interleave=params.interleave)
    fidelity = {}
    for target in targets:
        assert target in platform.calibration.two_qubits, (
            "Pair not calibrated, run standard 2q rb before interleaved version"
        )
        fidelity[target] = platform.calibration.two_qubits[target].rb_fidelity
    data.fidelity = fidelity
    return data


def _fit(data: RB2QInterData) -> StandardRB2QInterResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data: Data from the data acquisition stage.

    Returns:
        StandardRB2QInterResult: Aggregated and processed data.
    """

    qubits = data.pairs
    results = fit(qubits, data)
    fidelity_cz = {}
    for qubit in qubits:
        fid_cz = results.fidelity[qubit] / data.fidelity[qubit][0]
        # TODO: check this error formula
        uncertainty_cz = np.sqrt(
            1 / data.fidelity[qubit][0] ** 2 * results.fit_uncertainties[qubit][1] ** 2
            + (results.fidelity[qubit] / data.fidelity[qubit][0] ** 2) ** 2
            * data.fidelity[qubit][1] ** 2
        )
        fidelity_cz[qubit] = [fid_cz, uncertainty_cz]

    return StandardRB2QInterResult(
        results.fidelity,
        results.pulse_fidelity,
        results.fit_parameters,
        results.fit_uncertainties,
        results.error_bars,
        fidelity_cz,
    )


def _update(
    results: StandardRBResult, platform: CalibrationPlatform, target: QubitPairId
):
    """Write cz fidelity in calibration."""
    # TODO: shall we use the gate fidelity or the pulse fidelity
    target = tuple(target)
    platform.calibration.two_qubits[target].cz_fidelity = tuple(
        results.fidelity_cz[target]
    )


standard_rb_2q_inter = Routine(_acquisition, _fit, _plot, _update)
