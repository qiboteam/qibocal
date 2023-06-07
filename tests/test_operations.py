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
    qubits = {}

    for elem in task.qubits:
        if isinstance(elem, list):
            qubit_ids = []
            qubit_vals = []
            for qubit in elem:
                qubit_ids.append(qubit)
                qubit_vals.append(platform.qubits[qubit])
            qubits[tuple(qubit_ids)] = qubit_vals
        else:
            qubits[elem] = platform.qubits[elem]

    task.operation.acquisition(task.parameters, platform, qubits)
