import os
from pathlib import Path

import pytest
from qibolab._core.platform.load import PLATFORMS

from qibocal.calibration.platform import create_calibration_platform

TESTING_PLATFORM_NAMES = [
    "mock",
]


@pytest.fixture(autouse=True)
def cd(tmp_path_factory: pytest.TempdirFactory):
    path: Path = tmp_path_factory.mktemp("run")
    os.chdir(path)


@pytest.fixture(scope="module", params=TESTING_PLATFORM_NAMES)
def platform(request, monkeypatch):
    """Dummy platform to be used when there is no access to QPU."""
    monkeypatch.setenv(PLATFORMS, str(Path(__file__).parent / "platforms"))
    return create_calibration_platform(request.param)
