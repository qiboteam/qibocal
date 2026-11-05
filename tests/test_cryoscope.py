"""Tests for the cryoscope protocol."""

import numpy as np
import pytest
from conftest import PATH_TESTING_DATA

from qibocal.protocols.flux_dependence.cryoscope import (
    CryoscopeData,
    CryoscopeResults,
    _fit,
    _invert_arc,
)


# TODO: upload a data example.
@pytest.mark.skip()
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


def test_invert_arc():
    arc = np.poly1d([-6.0, 0.03, 0.0])
    for amplitude in (0.05, 0.2, 0.26):
        assert _invert_arc(arc, arc(amplitude), 0.2) == pytest.approx(amplitude)
        # the sign of the pulse selects the branch
        assert _invert_arc(arc, arc(-amplitude), -0.2) == pytest.approx(-amplitude)


def test_invert_arc_constant_offset():
    """The offset of the arc must not shift the amplitude."""
    arc = np.poly1d([-6.0, 0.03, 0.015])
    for amplitude in (0.01, 0.1, 0.25):
        detuning = arc(amplitude) - arc(0.0)
        assert _invert_arc(arc, detuning, 0.2) == pytest.approx(amplitude)


def test_invert_arc_out_of_range():
    """A detuning beyond the arc extremum returns the extremum."""
    arc = np.poly1d([-6.0, 0.0, 0.0])
    assert _invert_arc(arc, 1.0, 0.2) == pytest.approx(0.0)
