"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
from qibolab import create_platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/single_routines.yml"
RUNCARD = Runcard.load(PATH_TO_RUNCARD)
platform = create_platform("dummy")


@pytest.mark.parametrize("action", RUNCARD.actions)
def test_data_acquisition(action):
    """Test data acquisition for all routines using dummy"""
    task = Task(action)
    if task.operation.qubits_dependent:
        task.operation.acquisition(task.parameters, platform, platform.qubits)
    elif task.operation.platform_dependent:
        task.operation.acquisition(task.parameters, platform)
    else:
        task.operation.acquisition(task.parameters)
