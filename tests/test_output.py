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


# TODO: this is essentially a proto `qq run` invocation, it should be simplified as
# much as possible in the library, and made available in conftest
@pytest.fixture
def mock_output(tmp_path: Path, platform) -> tuple[Output, Path]:
    backend = construct_backend(backend="qibolab", platform="mock")
    platform: Platform = backend.platform
    meta = Metadata.generate(tmp_path.name, backend)
    output = Output(History(), meta, platform)
    platform.connect()
    meta.start()
    executor = Executor(
        history=History(), targets=list(platform.qubits), platform=platform
    )
    executor.run_protocol(flipping, ACTION, mode=ExecutionMode.ACQUIRE, output=tmp_path)
    meta.end()
    platform.disconnect()
    output.history = executor.history
    output.dump(tmp_path)

    return output, tmp_path


def test_output_process(mock_output: tuple[Output, Path]):
    """Create method of Executor."""
    output, path = mock_output
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
    assert path1.name.split("-")[3] == "000"
    assert path2.name.split("-")[3] == "001"


def test_output_mkdir():
    path1 = Output.mkdir()
    path2 = Output.mkdir()
    assert path1.name.split("-")[3] == "000"
    assert path2.name.split("-")[3] == "001"

    with pytest.raises(RuntimeError):
        Output.mkdir(path1)

    Output.mkdir(path1, force=True)
