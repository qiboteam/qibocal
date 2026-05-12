import pytest
from conftest import PATH_TESTING_DATA

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
    assert fitted.frequency[qubit] == pytest.approx(expected.frequency[qubit])
