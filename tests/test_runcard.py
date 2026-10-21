import pytest

from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import DummyRes, Protocol, dummy_operation
from qibocal.auto.runcard import Action, Runcard
from qibocal.auto.task import Completed, Task

EXAMPLE = {
    "targets": [0, 1],
    "actions": [
        {
            "id": "readout characterization",
            "operation": "readout_characterization",
            "parameters": {
                "nshots": 5000,
                "delay": 1000,
            },
        }
    ],
}


def test_load():
    ex = Runcard.load(EXAMPLE)

    assert ex.targets == [0, 1]
    assert len(ex.actions) == 1


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("runcard_update", [False, True])
@pytest.mark.parametrize("action_update", [False, True])
@pytest.mark.parametrize("fit", [False, True])
def test_run_update_controls(
    tmp_path, platform, mocker, update, runcard_update, action_update, fit
):
    protocol = Protocol(
        dummy_operation.acquisition, fit=mocker.Mock(), update=mocker.Mock()
    )
    mocker.patch("qibocal.protocols.test_protocol", protocol, create=True)
    action = Action("test", "test_protocol", update=action_update, parameters={})
    completed = Completed(Task(action, protocol), tmp_path, _results=DummyRes())
    run = mocker.patch.object(Task, "run", return_value=completed)
    apply_update = mocker.patch.object(completed, "update_platform")
    mode = ExecutionMode.ACQUIRE
    if fit:
        mode |= ExecutionMode.FIT
    runcard = Runcard(actions=[action], targets=[0], update=runcard_update)

    history = runcard.run(output=tmp_path, platform=platform, mode=mode, update=update)

    assert next(history.values()) is completed
    assert run.call_args.kwargs["platform"] is platform
    assert run.call_args.kwargs["targets"] == [0]
    assert run.call_args.kwargs["mode"] == mode
    assert run.call_args.kwargs["folder"] == tmp_path / "data" / "test-0"
    if fit and update and runcard_update and action_update:
        apply_update.assert_called_once_with(platform=platform)
    else:
        apply_update.assert_not_called()
