from pathlib import Path

import pytest

from qibocal.protocols.drag.drag import DragTuningData, DragTuningResults
from qibocal.protocols.drag.drag import _fit as drag_fit
from qibocal.protocols.drag.drag_simple import (
    DragTuningSimpleData,
    DragTuningSimpleResults,
)
from qibocal.protocols.drag.drag_simple import _fit as drag_simple_fit

TEST_FILE_DIR = Path(__file__).resolve().parent
PATH_TESTING_DATA = TEST_FILE_DIR / "tests_data"


def test_drag_fit():
    results_folder = PATH_TESTING_DATA / "drag_tuning-0"
    data = DragTuningData.load(results_folder)
    expected = DragTuningResults.load(results_folder)
    assert data is not None and expected is not None
    fitted = drag_fit(data)
    qubit = 0  # the data contains only qubit 0
    assert fitted.betas[qubit] == pytest.approx(expected.betas[qubit])
    assert fitted.fitted_parameters[qubit] == pytest.approx(
        expected.fitted_parameters[qubit]
    )
    # assert fitted.chi2[qubit] == pytest.approx(expected.chi2[qubit])


def test_drag_simple_fit():
    results_folder = PATH_TESTING_DATA / "drag_simple-0"
    data = DragTuningSimpleData.load(results_folder)
    expected = DragTuningSimpleResults.load(results_folder)
    assert data is not None and expected is not None
    fitted = drag_simple_fit(data)
    qubit = 0  # the data contains only qubit 0
    assert fitted.betas[qubit] == pytest.approx(expected.betas[qubit])
    assert fitted.fitted_parameters[(qubit, "YpX9")] == pytest.approx(
        expected.fitted_parameters[(qubit, "YpX9")]
    )
    assert fitted.fitted_parameters[(qubit, "XpY9")] == pytest.approx(
        expected.fitted_parameters[(qubit, "XpY9")]
    )
