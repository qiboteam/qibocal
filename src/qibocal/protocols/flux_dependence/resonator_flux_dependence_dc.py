from qibocal.calibration import CalibrationPlatform

from ...auto.operation import QubitId, Routine
from .resonator_flux_dependence import (
    ResonatorFluxData,
    ResonatorFluxParameters,
    ResonatorFluxResults,
    _acquisition_base,
    _fit,
    _plot,
    _update_base,
)

__all__ = ["resonator_flux_dc"]


def _acquisition(
    params: ResonatorFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorFluxData:
    return _acquisition_base(params, platform, targets, True)


def _update(
    results: ResonatorFluxResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    _update_base(results, platform, qubit, True)


resonator_flux_dc = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
