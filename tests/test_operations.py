"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
from qibolab import create_platform

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.utils import allocate_qubits_pairs, allocate_single_qubits

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/single_routines.yml"
RUNCARD = Runcard.load(PATH_TO_RUNCARD)
platform = create_platform("dummy")


@pytest.mark.parametrize("action", RUNCARD.actions)
def test_data_acquisition(action):
    """Test data acquisition for all routines using dummy"""
    task = Task(action)
    if any(isinstance(i, tuple) for i in task.qubits):
        qubits = allocate_qubits_pairs(platform, task.qubits)
    else:
        qubits = platform.qubits

    task.operation.acquisition(task.parameters, platform, qubits)
