import yaml
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

    action = yaml.safe_load((results_folder / "action.yml").read_text(encoding="utf-8"))
    [qubit] = action["targets"]

    fitted = resonator_spectroscopy_fit(data)
    assert fitted.frequency[qubit] == approx_for_regression(expected.frequency[qubit])
    assert fitted.fitted_parameters[qubit] == approx_for_regression(
        expected.fitted_parameters[qubit]
    )
    assert fitted.amplitude is not None and expected.amplitude is not None
    assert fitted.amplitude[qubit] == approx_for_regression(expected.amplitude[qubit])
