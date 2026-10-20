from dataclasses import dataclass, fields

import numpy as np

from qibocal.auto.operation import Protocol, QubitPairId
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

    results = fit(data, single_qubit=False)
    fidelity_cz = {}
    for qubit in data.pairs:
        # Interleaved RB fidielity as defined in Eq. (4) of
        # https://arxiv.org/abs/1203.4550.
        #
        # Only the reference fidelity F_ref = 1/4 + 3/4 * p_ref and its uncertainty are
        # stored in the calibration, so p_ref and sigma_p_ref are reconstructed by
        # inverting that relation. The reference and interleaved runs are assumed to be
        # statistically independent, so their errors are added in quadrature.
        f_ref, sigma_f_ref = data.fidelity[qubit]
        p_ref = (4 * f_ref - 1) / 3
        sigma_p_ref = 4 * sigma_f_ref / 3
        p_int = results.fit_parameters[qubit][1]
        sigma_p_int = results.fit_uncertainties[qubit][1]
        fid_cz = 1 / 4 + 3 / 4 * p_int / p_ref
        uncertainty_cz = (
            3
            / 4
            * np.sqrt(
                (sigma_p_int / p_ref) ** 2 + (p_int * sigma_p_ref / p_ref**2) ** 2
            )
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


standard_rb_2q_inter = Protocol(_acquisition, _fit, _plot, _update)
