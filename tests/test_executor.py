from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
from qibolab import Platform, create_platform

import qibocal
import qibocal.protocols
from qibocal import Executor
from qibocal.auto.history import History
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.auto.runcard import Action
from qibocal.calibration.platform import (
    create_calibration_platform,
)
from qibocal.protocols import flipping

PARAMETERS = {
    "id": "flipping",
    "targets": [0, 1],
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
def test_anonymous_executor(params, platform):
    """Executor without any name."""
    platform = (
        platform
        if isinstance(platform, Platform)
        else create_calibration_platform(platform)
    )
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

    init(path, force=True, platform="mock")
    assert executor.platform.name == "mock"

    init(path, force=True, platform=create_platform("mock"))
    assert executor.platform.name == "mock"


def test_close(tmp_path: Path, executor: Executor):
    path = tmp_path / "my-close-folder"

    init = executor.init
    close = executor.close

    init(path)
    close()

    assert executor.meta is not None
    assert executor.meta.start is not None
    assert executor.meta.end is not None


def test_context_manager(tmp_path: Path, executor: Executor):
    path = tmp_path / "my-ctx-folder"

    executor.init(path)

    with executor:
        assert executor.meta is not None
        assert executor.meta.start is not None


def test_open(tmp_path: Path):
    path = tmp_path / "my-open-folder"

    with Executor.open("myexec", path) as e:
        assert isinstance(e.t1, Callable)
        assert e.meta is not None
        assert e.meta.start is not None

    assert e.meta.end is not None
