import os
from copy import deepcopy
from dataclasses import dataclass
from importlib import reload
from pathlib import Path
from typing import Optional

import pytest
from qibolab import Platform, create_platform
from qibolab.qubits import QubitId

import qibocal
import qibocal.protocols
from qibocal import Executor
from qibocal.auto.history import History
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import Data, Parameters, Results, Routine
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
def test_anonymous_executor(params, platform):
    """Executor without any name."""
    platform = platform if isinstance(platform, Platform) else create_platform(platform)
    executor = Executor(
        history=History(),
        platform=platform,
        targets=list(platform.qubits),
        update=True,
    )
    executor.run_protocol(
        flipping, Action.cast(params, "flipping"), mode=ExecutionMode.ACQUIRE
    )

    assert executor.name is None


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
@pytest.mark.parametrize("platform", ["dummy", PLATFORM])
def test_named_executor(params, platform):
    """Create method of Executor."""
    executor = Executor.create("myexec", platform=platform)
    executor.run_protocol(flipping, Action.cast(params, "flipping"))
    executor.unload()


SCRIPTS = Path(__file__).parent / "scripts"


@dataclass
class FakeParameters(Parameters):
    par: int


@dataclass
class FakeData(Data):
    par: int


@dataclass
class FakeResults(Results):
    par: dict[QubitId, int]


def _acquisition(params: FakeParameters, platform) -> FakeData:
    return FakeData(par=params.par)


def _fit(data: FakeData) -> FakeResults:
    return FakeResults(par={0: data.par})


def _plot(data: FakeData, target: QubitId, fit: Optional[FakeResults] = None):
    pass


def _update(results: FakeResults, platform, qubit):
    pass


@pytest.fixture
def fake_protocols(request):
    marker = request.node.get_closest_marker("protocols")
    if marker is None:
        return

    protocols = {}
    for name in marker.args:
        routine = Routine(_acquisition, _fit, _plot, _update)
        setattr(qibocal.protocols, name, routine)
        protocols[name] = routine

    return protocols


@pytest.fixture
def executor():
    executor = Executor.create("my-exec")
    yield executor
    try:
        executor.unload()
    except KeyError:
        # it has been explicitly unloaded, no need to do it again
        pass


@pytest.mark.protocols("ciao", "come")
def test_simple(fake_protocols):
    globals_ = {}
    exec((SCRIPTS / "simple.py").read_text(), globals_)
    assert globals_["res"]._results.par[0] == 42


def test_init(tmp_path: Path, executor: Executor):
    path = tmp_path / "my-init-folder"

    init = executor.init

    init(path)
    with pytest.raises(RuntimeError, match="Directory .* already exists"):
        init(path)

    init(path, force=True)

    assert executor.meta is not None
    assert executor.meta.start is not None


def test_close(tmp_path: Path, executor: Executor):
    path = tmp_path / "my-close-folder"

    init = executor.init
    close = executor.close

    init(path)
    close()

    assert executor.meta is not None
    assert executor.meta.start is not None
    assert executor.meta.end is not None


def test_default_executor():
    os.environ["QIBO_PLATFORM"] = "ciao-come-va"
    reload(qibocal)
    assert qibocal.DEFAULT_EXECUTOR.platform.name == "dummy"
