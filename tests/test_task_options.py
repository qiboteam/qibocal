"""Test routines' acquisition method using dummy platform."""

from copy import deepcopy

import pytest
from pytest import approx

from qibocal import protocols
from qibocal.auto.mode import AUTOCALIBRATION, ExecutionMode
from qibocal.auto.operation import DEFAULT_PARENT_PARAMETERS
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Task
from qibocal.protocols.classification.classification import (
    SingleShotClassificationParameters,
)
from qibocal.protocols.readout.readout_mitigation_matrix import (
    ReadoutMitigationMatrixParameters,
)

TARGETS = [0, 1]
DUMMY_CARD = {
    "targets": TARGETS,
    "actions": [
        {
            "id": "standard rb",
            "operation": "standard_rb",
            "parameters": {
                "depths": [1, 5, 10],
                "niter": 3,
                "nshots": 10,
            },
        },
    ],
}


def modify_card(
    card, targets=None, local_update=None, global_update=None, backend="qibolab"
):
    """Modify runcard to change local targets or update."""
    card["backend"] = backend

    if global_update is not None:
        card["update"] = global_update

    for action in card["actions"]:
        if targets is not None:
            action["targets"] = targets
        elif local_update is not None:
            action["update"] = local_update
    return card


@pytest.mark.parametrize("backend", ["numpy", "qibolab"])
@pytest.mark.parametrize("local_targets", [None, [0, 1]])
def test_targets_argument(backend, local_targets, tmp_path):
    """Test possible qubits combinations between global and local."""
    runcard = Runcard.load(
        modify_card(DUMMY_CARD, targets=local_targets, backend=backend)
    )
    task = Task(runcard.actions[0], getattr(protocols, runcard.actions[0].operation))

    completed = task.run(
        platform=runcard.platform,
        targets=TARGETS,
        mode=ExecutionMode.ACQUIRE,
        folder=tmp_path,
    )
    if local_targets:
        assert completed.task.targets == local_targets
    else:
        assert completed.task.targets == TARGETS
        assert runcard.targets == TARGETS


UPDATE_CARD = {
    "targets": TARGETS,
    "actions": [
        {
            "id": "readout frequency",
            "operation": "single_shot_classification",
            "parameters": {"nshots": 100},
        },
    ],
}


# FIXME: handle local update
@pytest.mark.parametrize("global_update", [True, False])
@pytest.mark.parametrize("local_update", [True, False])
def test_update_argument(global_update, local_update, tmp_path, platform):
    """Test possible update combinations between global and local."""
    NEW_CARD = modify_card(
        UPDATE_CARD, local_update=local_update, global_update=global_update
    )
    old_excited_state = platform.calibration.single_qubits[0].readout.excited_state

    Runcard.load(NEW_CARD).run(
        tmp_path,
        mode=AUTOCALIBRATION,
        platform=platform,
    )

    if local_update and global_update:
        assert old_excited_state != approx(
            platform.calibration.single_qubits[0].readout.excited_state
        )

    else:
        assert old_excited_state == approx(
            platform.calibration.single_qubits[0].readout.excited_state
        )


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
