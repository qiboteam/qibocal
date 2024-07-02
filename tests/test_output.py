from copy import deepcopy
from pathlib import Path

import pytest
from qibo.backends import construct_backend
from qibolab import Platform

from qibocal import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.output import History, Metadata, Output
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
    meta = Metadata.generate(tmp_path.name, backend, "dummy")
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
