import pytest
from conftest import PATH_TESTING_DATA

from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    QubitFluxData,
    QubitFluxResults,
)
from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    _fit as qubit_flux_fit,
)

RESULT_FOLDERS = sorted(PATH_TESTING_DATA.glob("qubit_flux-*"))


@pytest.mark.parametrize("results_folder", [p for p in RESULT_FOLDERS])
def test_qubit_flux_fit(results_folder):
    """Regression test based on example experimental data"""
    data = QubitFluxData.load(results_folder)
    expected = QubitFluxResults.load(results_folder)

    assert data is not None and expected is not None

    fitted = qubit_flux_fit(data)

    for qubit in expected.frequency:
        assert (
            pytest.approx(expected.frequency[qubit], abs=1e3) == fitted.frequency[qubit]
        )
        assert (
            pytest.approx(expected.sweetspot[qubit], abs=1e-3)
            == fitted.sweetspot[qubit]
        )
