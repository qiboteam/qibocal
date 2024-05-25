import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from qibolab import create_platform
from qibolab.qubits import QubitId

import qibocal.protocols
from qibocal import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import Parameters, Results, Routine
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


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
@pytest.mark.parametrize("platform", ["dummy", PLATFORM])
def test_named_executor(params, platform, tmp_path):
    """Create method of Executor."""
    executor = Executor.create("myexec", platform=platform, output=tmp_path)
    executor.run_protocol(flipping, params)
    del sys.modules["myexec"]


SCRIPTS = Path(__file__).parent / "scripts"


@dataclass
class FakeParameters(Parameters):
    par: int


@dataclass
class FakeResults(Results):
    par: dict[QubitId, int]


def _acquisition(params: FakeParameters, platform):
    return FakeResults({0: params.par})


def _update(results, platform, qubit):
    pass


@pytest.fixture
def fake_protocols(request):
    marker = request.node.get_closest_marker("protocols")
    if marker is None:
        return

    protocols = {}
    for name in marker.args:
        routine = Routine(_acquisition, lambda x: x, lambda x: x, _update)
        setattr(qibocal.protocols, name, routine)
        protocols[name] = routine

    return protocols


# class TestScripts:
@pytest.mark.protocols("ciao", "come")
def test_simple(fake_protocols):
    globals_ = {}
    exec((SCRIPTS / "simple.py").read_text(), globals_)
    assert globals_["res"]._results.par[0] == 42
