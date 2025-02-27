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


def set_platform_profile():
    os.environ[PLATFORMS] = str(Path(__file__).parent / "platforms")


@pytest.fixture(scope="module", params=TESTING_PLATFORM_NAMES)
def platform(request):
    """Dummy platform to be used when there is no access to QPU."""
    set_platform_profile()
    return create_calibration_platform(request.param)
