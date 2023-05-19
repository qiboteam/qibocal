"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
from qibolab.platform import Platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/single_routines.yml"
RUNCARD = Runcard.load(PATH_TO_RUNCARD)
platform = Platform("dummy")


@pytest.mark.parametrize("action", RUNCARD.actions)
def test_data_acquisition(action):
    """Test data acquisition for all routines using dummy"""
    task = Task(action)
    if "low attenuation" in task.id or "high attenuation" in task.id:
        with pytest.raises(NotImplementedError):
            task.operation.acquisition(task.parameters, platform, platform.qubits)
    else:
        task.operation.acquisition(task.parameters, platform, platform.qubits)
