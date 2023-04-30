"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
from qibolab.platform import Platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task

card = pathlib.Path(__file__).parent / "runcards"
runcard = Runcard.load(card / "dummy_autocalibration.yml")
platform = Platform("dummy")


@pytest.mark.parametrize("action", runcard.actions)
def test_data_acquisition(action):
    task = Task(action)
    print(task)
    if "low attenuation" in task.id or "high attenuation" in task.id and platform.name:
        with pytest.raises(NotImplementedError):
            task.operation.acquisition(task.parameters, platform, platform.qubits)
    else:
        task.operation.acquisition(task.parameters, platform, platform.qubits)
