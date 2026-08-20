"""Tests for the cryoscope protocol."""

import pytest
from conftest import PATH_TESTING_DATA

from qibocal.protocols.flux_dependence.cryoscope import (
    CryoscopeData,
    CryoscopeResults,
    _fit,
)


def test_cryoscope_fit():
    results_folder = PATH_TESTING_DATA / "cryoscope-0"
    data = CryoscopeData.load(results_folder)
    assert data is not None
    expected = CryoscopeResults.load(results_folder)
    assert expected is not None
    fitted = _fit(data)

    qubit = 0  # the data contains only qubit 0
    assert fitted.fir_taps[qubit] == pytest.approx(expected.fir_taps[qubit])
    assert fitted.exp_amplitude[qubit] == pytest.approx(expected.exp_amplitude[qubit])
    assert fitted.tau[qubit] == pytest.approx(expected.tau[qubit])
