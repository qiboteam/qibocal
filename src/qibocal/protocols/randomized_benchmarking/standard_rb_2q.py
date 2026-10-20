from dataclasses import dataclass

import plotly.graph_objects as go

from qibocal.auto.operation import Protocol, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    survival_figure,
)

from ...calibration.calibration import TwoQubitCalibration
from ..utils import table_dict, table_html
from .utils import (
    RB2QData,
    StandardRBResult,
    fidelity_uncertainty,
    fit,
    twoq_rb_acquisition,
)

__all__ = ["StandardRB2QParameters", "standard_rb_2q"]


FILE_CLIFFORDS = "2qubitCliffs.json"
FILE_INV = "2qubitCliffsInv.npz"
TWO_QUBIT_DIMENSION = 2**2
"""Hilbert space dimension of a qubit pair."""


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
    return fit(data, single_qubit=False)


def metadata_table(data: RB2QData, target: QubitPairId) -> str:
    """Table with the acquisition parameters of a two qubit RB experiment."""
    return table_html(
        table_dict(
            str(target),
            ["niter", "nshots"],
            [data.niter, data.nshots],
        )
    )


def _plot(
    data: RB2QData, fit: StandardRBResult | None, target: QubitPairId
) -> tuple[list[go.Figure], str]:
    """Survival probability figure and two qubit Clifford fidelity table."""
    fig = survival_figure(data, fit, target, single_qubit=False)
    if fit is None:
        return [fig], ""

    sigma_decay = fit.fit_uncertainties[target][1]
    sigma_fidelity = fidelity_uncertainty(sigma_decay, TWO_QUBIT_DIMENSION)
    fitting_report = table_html(
        table_dict(
            str(target),
            ["F_Cl"],
            [(fit.fidelity[target], sigma_fidelity)],
            display_error=True,
        )
    )
    model_note = "<p>F_Cl = (1 + 3p) / 4</p>"
    return [fig], metadata_table(data, target) + fitting_report + model_note


def _update(
    results: StandardRBResult, platform: CalibrationPlatform, target: QubitPairId
):
    """Write rb fidelity in calibration."""
    if target not in platform.calibration.two_qubits:
        platform.calibration.two_qubits[target] = TwoQubitCalibration()

    platform.calibration.two_qubits[target].rb_fidelity = (
        results.fidelity[target],
        fidelity_uncertainty(results.fit_uncertainties[target][1], TWO_QUBIT_DIMENSION),
    )


standard_rb_2q = Protocol(_acquisition, _fit, _plot, _update)
