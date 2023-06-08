"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile

import pytest
from qibolab import create_platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.utils import allocate_qubits

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/single_routines.yml"
RUNCARD = Runcard.load(PATH_TO_RUNCARD)
platform = create_platform("dummy")


@pytest.mark.parametrize("global_qubits", [[], [0, 1]])
@pytest.mark.parametrize("local_qubits", [[], [0, 1]])
@pytest.mark.parametrize("action", RUNCARD.actions)
def test_qubits_behavior(action, global_qubits, local_qubits):
    """Test data acquisition for all routines using dummy"""
    action.qubits = local_qubits
    task = Task(action)
    if not local_qubits and not global_qubits:
        with pytest.raises(ValueError):
            task.run(
                pathlib.Path(tempfile.mkdtemp()),
                platform,
                allocate_qubits(platform, global_qubits),
            )
    else:
        task.run(
            pathlib.Path(tempfile.mkdtemp()),
            platform,
            allocate_qubits(platform, global_qubits),
        )
        if local_qubits:
            assert task.qubits == local_qubits
        else:
            assert task.qubits == global_qubits
