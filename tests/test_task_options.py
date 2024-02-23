"""Test routines' acquisition method using dummy platform"""

from copy import deepcopy

import pytest
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.operation import DEFAULT_PARENT_PARAMETERS
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.cli.report import ExecutionMode
from qibocal.protocols.characterization.classification import (
    SingleShotClassificationParameters,
)
from qibocal.protocols.characterization.readout_mitigation_matrix import (
    ReadoutMitigationMatrixParameters,
)

PLATFORM = create_platform("dummy")
QUBITS = list(PLATFORM.qubits)
DUMMY_CARD = {
    "backend": "numpy",
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


def modify_card(card, targets=None, update=None):
    """Modify runcard to change local targets or update."""
    for action in card["actions"]:
        if targets is not None:
            action["targets"] = targets
        elif update is not None:
            action["update"] = update
    return card


@pytest.mark.parametrize("platform", [None, PLATFORM])
@pytest.mark.parametrize("local_targets", [None, [0, 1]])
def test_qubits_argument(platform, local_targets, tmp_path):
    """Test possible qubits combinations between global and local."""
    runcard = Runcard.load(modify_card(DUMMY_CARD, targets=local_targets))
    print(runcard)
    task = Task(runcard.actions[0])

    completed = task.run(
        max_iterations=1,
        platform=platform,
        targets=list(QUBITS),
        mode=ExecutionMode.acquire,
        folder=tmp_path,
    )
    if local_targets:
        assert completed.task.targets == local_targets
    else:
        assert completed.task.targets == list(QUBITS)


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
def test_update_argument(global_update, local_update, tmp_path):
    """Test possible update combinations between global and local."""
    platform = create_platform("dummy")
    NEW_CARD = modify_card(UPDATE_CARD, update=local_update)
    executor = Executor.load(
        Runcard.load(NEW_CARD),
        tmp_path,
        platform,
        platform.qubits,
        update=global_update,
    )

    old_readout_frequency = executor.platform.qubits[0].readout_frequency
    old_iq_angle = executor.platform.qubits[1].iq_angle

    list(executor.run(mode=ExecutionMode.autocalibration))

    if local_update and global_update:
        assert old_readout_frequency != executor.platform.qubits[0].readout_frequency
        assert old_iq_angle != executor.platform.qubits[1].iq_angle

    else:
        assert old_readout_frequency == executor.platform.qubits[0].readout_frequency
        assert old_iq_angle == executor.platform.qubits[1].iq_angle


@pytest.mark.parametrize(
    "ChildClass",
    [SingleShotClassificationParameters, ReadoutMitigationMatrixParameters],
)
@pytest.mark.parametrize("use_parameters", [True, False])
def test_parameters_load(ChildClass, use_parameters):
    if use_parameters:
        parameters = {key: 10 for key in DEFAULT_PARENT_PARAMETERS.keys()}
    else:
        parameters = {}
    parameters2 = deepcopy(parameters)
    child_class = ChildClass().load(parameters)
    for parameter, value in DEFAULT_PARENT_PARAMETERS.items():
        if parameter in parameters2:
            assert getattr(child_class, parameter) == parameters2[parameter]
        else:
            assert getattr(child_class, parameter) == value
