import pytest
from conftest import PATH_TESTING_DATA

from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    ResonatorFluxData,
    ResonatorFluxResults,
)
from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    _fit as resonator_flux_fit,
)

RESULT_FOLDERS = sorted(PATH_TESTING_DATA.glob("resonator_flux-*"))


@pytest.mark.parametrize("results_folder", [p for p in RESULT_FOLDERS])
def test_resonator_flux_fit(results_folder):
    """Regression test based on example experimental data"""
    data = ResonatorFluxData.load(results_folder)
    expected = ResonatorFluxResults.load(results_folder)

    assert data is not None and expected is not None

    fitted = resonator_flux_fit(data)

    for qubit in expected.frequency:
        assert (
            pytest.approx(expected.frequency[qubit], abs=20e3)
            == fitted.frequency[qubit]
        )
        assert (
            pytest.approx(expected.sweetspot[qubit], abs=1e-3)
            == fitted.sweetspot[qubit]
        )
