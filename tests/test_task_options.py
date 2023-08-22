"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile
from copy import deepcopy

import pytest
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.cli.builders import ExecutionMode
from qibocal.utils import allocate_single_qubits

PLATFORM = create_platform("dummy")
QUBITS = list(PLATFORM.qubits)
DUMMY_CARD = {
    "qubits": QUBITS,
    "actions": [
        {
            "id": "standard rb",
            "priority": 0,
            "operation": "standard_rb",
            "parameters": {
                "depths": [1, 5, 10],
                "niter": 3,
                "nshots": 10,
            },
        },
    ],
}


def modify_card(card, qubits=None, update=None):
    """Modify runcard to change local qubits or update."""
    for action in card["actions"]:
        if qubits is not None:
            action["qubits"] = qubits
        elif update is not None:
            action["update"] = update
    return card


@pytest.mark.parametrize("platform", [None, PLATFORM])
@pytest.mark.parametrize("local_qubits", [[], [0, 1]])
def test_qubits_argument(platform, local_qubits):
    """Test possible qubits combinations between global and local."""
    runcard = Runcard.load(modify_card(DUMMY_CARD, qubits=local_qubits))
    task = Task(runcard.actions[0])
    global_qubits = (
        allocate_single_qubits(platform, QUBITS) if platform is not None else QUBITS
    )
    execution = task.run(platform, global_qubits)
    data = next(execution)
    if local_qubits:
        assert task.qubits == local_qubits
    else:
        assert task.qubits == QUBITS


UPDATE_CARD = {
    "qubits": QUBITS,
    "actions": [
        {
            "id": "resonator frequency",
            "priority": 0,
            "operation": "resonator_spectroscopy",
            "main": "classification",
            "parameters": {
                "freq_width": 10_000_000,
                "freq_step": 100_000,
                "amplitude": 0.4,
                "power_level": "low",
            },
        },
        {
            "id": "classification",
            "priority": 0,
            "operation": "single_shot_classification",
            "parameters": {"nshots": 100},
        },
    ],
}


@pytest.mark.parametrize("global_update", [True, False])
@pytest.mark.parametrize("local_update", [True, False])
def test_update_argument(global_update, local_update):
    """Test possible update combinations between global and local."""
    platform = deepcopy(create_platform("dummy"))
    old_readout_frequency = platform.qubits[0].readout_frequency
    old_iq_angle = platform.qubits[1].iq_angle
    NEW_CARD = modify_card(deepcopy(UPDATE_CARD), update=local_update)
    executor = Executor.load(
        Runcard.load(NEW_CARD),
        pathlib.Path(tempfile.mkdtemp()),
        platform,
        platform.qubits,
        global_update,
    )

    for _ in executor.run(mode=ExecutionMode.autocalibration):
        pass

    if local_update and global_update:
        assert old_readout_frequency != platform.qubits[0].readout_frequency
        assert old_iq_angle != platform.qubits[1].iq_angle

    else:
        assert old_readout_frequency == platform.qubits[0].readout_frequency
        assert old_iq_angle == platform.qubits[1].iq_angle
