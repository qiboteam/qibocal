from copy import deepcopy
from pathlib import Path

import pytest
from qibo.backends import construct_backend
from qibolab import Platform

from qibocal import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.output import History, Metadata, Output, TaskStats, _new_output
from qibocal.auto.runcard import Action
from qibocal.protocols import flipping

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


# TODO: this is essentially a proto `qq auto` invocation, it should be simplified as
# much as possible in the library, and made available in conftest
@pytest.fixture
def fake_output(tmp_path: Path) -> tuple[Output, Path]:
    backend = construct_backend(backend="qibolab", platform="dummy")
    platform: Platform = backend.platform
    meta = Metadata.generate(tmp_path.name, backend)
    output = Output(History(), meta, platform)
    platform.connect()
    meta.start()
    executor = Executor.create(platform=platform)
    executor.run_protocol(flipping, ACTION, mode=ExecutionMode.ACQUIRE)
    meta.end()
    platform.disconnect()
    output.history = executor.history
    output.dump(tmp_path)

    return output, tmp_path


def test_output_process(fake_output: tuple[Output, Path]):
    """Create method of Executor."""
    output, path = fake_output
    # perform fit
    output.process(path, mode=ExecutionMode.FIT)

    # check double fit error
    with pytest.raises(KeyError):
        output.process(path, mode=ExecutionMode.FIT)


def test_task_stats():
    stats = TaskStats(2, 5)
    assert stats.fit == 5
    assert stats.tot == 7


def test_new_output():
    path1 = _new_output()
    path1.mkdir()
    path2 = _new_output()

    assert str(path1).split("-")[-2] == "000"
    assert str(path2).split("-")[-2] == "001"


def test_output_mkdir():
    path1 = Output.mkdir()
    path2 = Output.mkdir()

    assert str(path1).split("-")[-2] == "000"
    assert str(path2).split("-")[-2] == "001"

    with pytest.raises(RuntimeError):
        Output.mkdir(path1)

    Output.mkdir(path1, force=True)
