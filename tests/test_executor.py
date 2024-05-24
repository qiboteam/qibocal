from copy import deepcopy
from pathlib import Path

import pytest
from qibolab import create_platform

import qibocal.protocols
from qibocal import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.runcard import Action
from qibocal.protocols import flipping

PLATFORM = create_platform("dummy")
PARAMETERS = {
    "id": "flipping",
    "targets": [0, 1, 2],
    "parameters": {
        "nflips_max": 20,
        "nflips_step": 2,
        "delta_amplitude": +0.1,
    },
}
action = deepcopy(PARAMETERS)
action["operation"] = "flipping"
ACTION = Action(**action)


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
@pytest.mark.parametrize("platform", ["dummy", PLATFORM])
def test_executor_create(params, platform):
    """Create method of Executor."""
    executor = Executor.create(platform=platform)
    executor.run_protocol(flipping, params, mode=ExecutionMode.ACQUIRE)


SCRIPTS = Path(__file__).parent / "scripts"


@pytest.fixture
def fake_protocols(request):
    marker = request.node.get_closest_marker("protocols")
    if marker is None:
        return

    protocols = {}
    for name in marker.args:

        def routine(args=name):
            return args

        setattr(qibocal.protocols, name, routine)
        protocols[name] = routine

    return protocols


# class TestScripts:
@pytest.mark.protocols("ciao", "come")
def test_simple(fake_protocols):
    globals_ = {}
    exec((SCRIPTS / "simple.py").read_text(), globals_)
    assert globals_["out"] == "ciao"
