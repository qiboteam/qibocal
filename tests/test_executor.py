import pytest
from qibolab import create_platform

from qibocal import Executor
from qibocal.auto.runcard import Action

PLATFORM = create_platform("dummy")
PROTOCOL = {
    "id": "flipping",
    "operation": "flipping",
    "targets": [0, 1, 2],
    "parameters": {
        "nflips_max": 20,
        "nflips_step": 2,
        "detuning": +0.1,
    },
}


@pytest.mark.parametrize("protocol", [PROTOCOL, Action(**PROTOCOL)])
@pytest.mark.parametrize("platform", ["dummy", PLATFORM])
def test_executor_create(platform, protocol, tmp_path):
    """Create method of Executor"""
    executor = Executor.create(protocols=[protocol], platform=platform, output=tmp_path)
    executor.run_protocol("flipping")
