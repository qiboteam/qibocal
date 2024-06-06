from copy import deepcopy

import pytest
from qibolab import create_platform

from qibocal import Executor
from qibocal.auto.runcard import Action
from qibocal.protocols import flipping

PLATFORM = create_platform("dummy")
PARAMETERS = {
    "id": "flipping",
    "targets": [0, 1, 2],
    "parameters": {
        "nflips_max": 20,
        "nflips_step": 2,
        "detuning": +0.1,
    },
}
action = deepcopy(PARAMETERS)
action["operation"] = "flipping"
ACTION = Action(**action)


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
@pytest.mark.parametrize("platform", ["dummy", PLATFORM])
def test_executor_create(params, platform, tmp_path):
    """Create method of Executor"""
    executor = Executor.create(platform=platform, output=tmp_path)
    executor.run_protocol(flipping, params)
