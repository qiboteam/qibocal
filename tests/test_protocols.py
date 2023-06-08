"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile

import pytest
from qibolab import create_platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/single_routines.yml"
RUNCARD = Runcard.load(PATH_TO_RUNCARD)
platform = create_platform("dummy")


@pytest.mark.parametrize("action", RUNCARD.actions)
def test_protocols(action):
    """Test data acquisition for all routines using dummy"""
    task = Task(action)
    task.run(pathlib.Path(tempfile.mkdtemp()), platform, platform.qubits)
