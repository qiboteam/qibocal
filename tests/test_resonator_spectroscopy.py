from conftest import PATH_TESTING_DATA, approx_for_regression

from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
    ResonatorSpectroscopyResults,
)
from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    _fit as resonator_spectroscopy_fit,
)


def test_resonator_spectroscopy_fit():
    results_folder = PATH_TESTING_DATA / "resonator_spectroscopy-0"
    data = ResonatorSpectroscopyData.load(results_folder)
    expected = ResonatorSpectroscopyResults.load(results_folder)
    assert data is not None and expected is not None
    fitted = resonator_spectroscopy_fit(data)
    qubit = 0  # the data contains only qubit 0
    assert fitted.frequency[qubit] == approx_for_regression(expected.frequency[qubit])
    assert fitted.fitted_parameters[qubit] == approx_for_regression(
        expected.fitted_parameters[qubit]
    )
    assert fitted.error_fit_pars[qubit] == approx_for_regression(
        expected.error_fit_pars[qubit]
    )
    assert fitted.chi2_reduced[qubit] == approx_for_regression(
        expected.chi2_reduced[qubit]
    )
    assert fitted.amplitude is not None and expected.amplitude is not None
    assert fitted.amplitude[qubit] == approx_for_regression(expected.amplitude[qubit])
