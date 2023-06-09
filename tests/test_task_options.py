"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile
from copy import deepcopy

import pytest
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.utils import allocate_qubits

PLATFORM = create_platform("dummy")
QUBITS = list(PLATFORM.qubits)

DUMMY_CARD = {
    "qubits": QUBITS,
    "actions": [
        {
            "id": "resonator frequency",
            "priority": 0,
            "operation": "resonator_spectroscopy",
            "parameters": {
                "freq_width": 10_000_000,
                "freq_step": 100_000,
                "amplitude": 0.4,
                "power_level": "low",
            },
        }
    ],
}


def modify_card(card, qubits=None, update=None):
    """Modify runcard to change local qubits or update."""
    if qubits is not None:
        card["actions"][0]["qubits"] = qubits
    elif update is not None:
        card["actions"][0]["update"] = update
    return card


@pytest.mark.parametrize("local_qubits", [[], [0, 1]])
def test_qubits_argument(local_qubits):
    """Test possible qubits combinations between global and local."""
    runcard = Runcard.load(modify_card(DUMMY_CARD, qubits=local_qubits))
    task = Task(runcard.actions[0])

    task.run(
        pathlib.Path(tempfile.mkdtemp()),
        PLATFORM,
        allocate_qubits(PLATFORM, QUBITS),
    )
    if local_qubits:
        assert task.qubits == local_qubits
    else:
        assert task.qubits == QUBITS


@pytest.mark.parametrize("global_update", [True, False])
@pytest.mark.parametrize("local_update", [True, False])
def test_update_argument(global_update, local_update):
    """Test possible update combinations between global and local."""
    platform = deepcopy(create_platform("dummy"))
    old_readout_frequency = platform.qubits[0].readout_frequency
    NEW_CARD = modify_card(deepcopy(DUMMY_CARD), update=local_update)
    executor = Executor.load(
        Runcard.load(NEW_CARD),
        pathlib.Path(tempfile.mkdtemp()),
        platform,
        platform.qubits,
        global_update,
    )
    executor.run()
    if local_update and global_update:
        assert old_readout_frequency != platform.qubits[0].readout_frequency
    else:
        assert old_readout_frequency == platform.qubits[0].readout_frequency
