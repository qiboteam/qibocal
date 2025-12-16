from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform

from .qubit_flux_dependence import (
    QubitFluxData,
    QubitFluxParameters,
    QubitFluxResults,
    _acquisition_base,
    _fit,
    _plot,
    _update_base,
)

__all__ = ["qubit_flux_dc"]


def _acquisition(
    params: QubitFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitFluxData:
    return _acquisition_base(params, platform, targets, True)


def _update(
    results: QubitFluxResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    _update_base(results, platform, qubit, True)


qubit_flux_dc = Routine(_acquisition, _fit, _plot, _update)
"""QubitFlux Routine object."""
