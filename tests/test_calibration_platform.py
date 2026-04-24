from pathlib import Path

import pytest
from qibolab import create_platform
from qibolab._core.platform.load import PLATFORMS

from qibocal.calibration.platform import CalibrationError, CalibrationPlatform


def test_validation_calibration_platform(monkeypatch):
    """Test for phase validation in the CalibrationPlatform initialization."""

    monkeypatch.setenv(PLATFORMS, str(Path(__file__).parent / "platforms"))

    faulty_plat_name = "mock1_faulty"
    faulty_platform = create_platform(faulty_plat_name)
    with pytest.raises(CalibrationError):
        _ = CalibrationPlatform.from_platform(faulty_platform)

    good_plat_name = "mock2"
    good_platform = create_platform(good_plat_name)
    cal_plat = CalibrationPlatform.from_platform(good_platform)
    assert isinstance(cal_plat, CalibrationPlatform)
