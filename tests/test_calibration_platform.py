from pathlib import Path

import numpy as np
import pytest
from qibolab import create_platform
from qibolab._core.platform.load import PLATFORMS_PATH

from qibocal.calibration.platform import CalibrationError, CalibrationPlatform


def test_validation_calibration_platform(monkeypatch):
    """Test for phase validation in the CalibrationPlatform initialization."""

    monkeypatch.setenv(PLATFORMS_PATH, str(Path(__file__).parent / "platforms"))

    faulty_plat_name = "mock1_faulty"
    faulty_platform = create_platform(faulty_plat_name)
    with pytest.raises(CalibrationError):
        _ = CalibrationPlatform.from_platform(faulty_platform)

    good_plat_name = "mock2"
    good_platform = create_platform(good_plat_name)
    cal_plat = CalibrationPlatform.from_platform(good_platform)
    assert isinstance(cal_plat, CalibrationPlatform)


def test_from_datafolder_does_not_load_hardware_platform(
    tmp_path, platform, monkeypatch
):
    # Save the platform snapshot used to reconstruct calibration state without
    # requiring the original platform's hardware objects to be created again.
    platform.dump(tmp_path)

    # Fail immediately if reconstruction tries to load the named platform from
    # the hardware/platform registry instead of using the saved data folder.
    def fail_if_platform_is_created(_platform_name):
        pytest.fail("from_datafolder must not create a hardware platform")

    monkeypatch.setattr(
        "qibocal.calibration.platform.create_platform", fail_if_platform_is_created
    )

    reconstructed = CalibrationPlatform.from_datafolder(tmp_path, platform.name)

    # The platform parameters and calibration contents needed for fitting must
    # survive the dump/reload round trip.
    assert reconstructed.parameters == platform.parameters

    assert reconstructed.calibration is not None
    assert platform.calibration is not None

    assert reconstructed.calibration.single_qubits == platform.calibration.single_qubits

    # Compare sparse matrix values explicitly: sparse-matrix == returns a sparse
    # result whose truth value is ambiguous, rather than a single boolean.
    reconstructed_matrix = reconstructed.calibration.readout_mitigation_matrix
    platform_matrix = platform.calibration.readout_mitigation_matrix
    assert reconstructed_matrix is not None
    assert platform_matrix is not None
    # Sparse matrix equality is ambiguous, so compare the underlying values.
    np.testing.assert_array_equal(
        reconstructed_matrix.toarray(),
        platform_matrix.toarray(),
    )

    # Calibration qubit identifiers remain available, but the reconstructed
    # object must not retain hardware instruments/couplers or claim a connection.
    assert list(reconstructed.qubits) == platform.calibration.qubits
    assert reconstructed.instruments == {}
    assert reconstructed.couplers == {}
    assert not reconstructed.is_connected
