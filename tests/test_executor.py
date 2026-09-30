from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import pytest
from qibolab import Platform

import qibocal
import qibocal.protocols
from qibocal import Executor
from qibocal.auto.execute import check_overlap_in_input_qubits
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import (
    Completed,
    Data,
    Parameters,
    Protocol,
    QubitId,
    Results,
)
from qibocal.auto.output import PLATFORM, Output
from qibocal.auto.runcard import Action
from qibocal.auto.task import Task
from qibocal.calibration.platform import (
    CalibrationPlatform,
    create_calibration_platform,
)
from qibocal.protocols import flipping

PARAMETERS = {
    "id": "flipping",
    "targets": [0, 1],
    "parameters": {
        "nflips_max": 20,
        "nflips_step": 2,
        "delta_amplitude": +0.1,
    },
}
action = deepcopy(PARAMETERS)
action["operation"] = "flipping"
ACTION = Action(**action)


@pytest.fixture
def bound_protocol(mocker):
    protocol = Protocol(
        acquisition=mocker.Mock(return_value=7),
        fit=mocker.Mock(return_value=11),
        report=mocker.Mock(),
        update=mocker.Mock(),
    )
    return protocol(pars=3, fit={"fit": True}, report={"report": True})


def test_bound_protocol_executor(bound_protocol, platform):
    executor = Executor(platform)
    completed = executor(bound_protocol)

    assert isinstance(completed, Completed)
    assert completed.data == 7
    assert completed.results == 11
    assert completed.success
    bound_protocol.protocol.acquisition.assert_called_once_with(3)
    bound_protocol.protocol.fit.assert_called_once_with(7, {"fit": True})
    bound_protocol.protocol.report.assert_called_once_with(7, 11, {"report": True})
    bound_protocol.protocol.update.assert_called_once_with(11, platform)
    assert executor.path is None
    assert list(executor.history) == []
    assert not hasattr(qibocal, "CalibrationExecutor")


@pytest.mark.parametrize("skip_fit", [False, True])
def test_bound_protocol_without_fit(bound_protocol, platform, skip_fit):
    fit = bound_protocol.protocol.fit
    if not skip_fit:
        bound_protocol.protocol.fit = None
    completed = Executor(platform)(bound_protocol, skip_fit=skip_fit)

    assert completed.data == 7
    assert completed.results is None
    fit.assert_not_called()
    bound_protocol.protocol.report.assert_not_called()
    bound_protocol.protocol.update.assert_not_called()


def test_bound_protocol_optional_report_and_update(bound_protocol, platform):
    bound_protocol.protocol.report = None
    bound_protocol.protocol.update = None
    completed = Executor(platform)(bound_protocol)

    assert completed.results == 11


def test_bound_protocol_update_disabled(bound_protocol, platform):
    executor = Executor(platform, update=False)
    completed = executor(bound_protocol)

    bound_protocol.protocol.update.assert_not_called()
    executor.update(completed.results, bound_protocol)
    bound_protocol.protocol.update.assert_called_once_with(11, platform)


def test_bound_protocol_individual_phases(bound_protocol, platform):
    executor = Executor(platform)
    data = executor.acquire(bound_protocol)
    results = executor.fit(data, bound_protocol)
    executor.report(data, results, bound_protocol)
    executor.update(results, bound_protocol)

    bound_protocol.protocol.acquisition.assert_called_once_with(3)
    bound_protocol.protocol.fit.assert_called_once_with(7, {"fit": True})
    bound_protocol.protocol.report.assert_called_once_with(7, 11, {"report": True})
    bound_protocol.protocol.update.assert_called_once_with(11, platform)


@pytest.mark.parametrize("phase", ["fit", "report", "update"])
def test_bound_protocol_missing_phase(bound_protocol, platform, phase):
    setattr(bound_protocol.protocol, phase, None)
    executor = Executor(platform)
    args = {
        "fit": (7, bound_protocol),
        "report": (7, 11, bound_protocol),
        "update": (11, bound_protocol),
    }

    with pytest.raises(ValueError, match="Protocol does not support"):
        getattr(executor, phase)(*args[phase])


def test_bound_protocol_update_requires_platform(bound_protocol):
    with pytest.raises(ValueError, match="does not have a platform"):
        Executor(None).update(11, bound_protocol)


@pytest.mark.parametrize("phase", ["acquisition", "fit", "report", "update"])
def test_bound_protocol_propagates_errors(bound_protocol, platform, phase):
    getattr(bound_protocol.protocol, phase).side_effect = RuntimeError(phase)

    with pytest.raises(RuntimeError, match=phase):
        Executor(platform)(bound_protocol)


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
def test_executor(params: dict | Action, platform: Platform | str, tmp_path: Path):
    """Executor without any name."""
    platform = (
        platform
        if isinstance(platform, Platform)
        else create_calibration_platform(platform)
    )
    executor = Executor.create(
        platform=platform,
        targets=list(platform.qubits),
        update=True,
        path=tmp_path,
    )
    executor.run_protocol(
        flipping, Action.cast(params, "flipping"), mode=ExecutionMode.ACQUIRE
    )


def test_executor_fit_reconstructs_platform_from_datafolder(
    executor: Executor, monkeypatch
):
    """Verifies that when the executor runs a FIT-only protocol, it reconstructs
    the platform from the serialized data folder (rather than reusing the live hardware
    platform) and passes that reconstructed platform to Task.run.
    """
    # FIT-only execution reconstructs its platform from this serialized folder.
    platform_folder = executor.path / PLATFORM
    platform_folder.mkdir(parents=True)
    executor.platform.dump(platform_folder)

    action = deepcopy(ACTION)
    # Disable update so the ACQUIRE run doesn't mutate the platform's calibration,
    # keeping the saved snapshot identical to what FIT will reconstruct.
    action.update = False
    executor.run_protocol(flipping, action, mode=ExecutionMode.ACQUIRE)

    acquired_folder = executor.history.task_path(
        executor.history._executed_task_id(action.id), executor.path
    )
    fit_folder = executor.history.task_path(
        executor.history._pending_task_id(action.id), executor.path
    )
    fit_folder.mkdir(parents=True)
    # FIT creates the next task iteration, and Task.run loads data from that folder.
    # Copy the acquisition payload there so the FIT call can run without reacquiring.
    for data_file in acquired_folder.glob("data.*"):
        copy2(data_file, fit_folder)

    observed_platforms = []
    # Save the original before patching: the wrapper records Executor's platform choice,
    # then delegates to Task.run so the normal data loading and fit still happen.
    task_run = Task.run

    def observe_platform(self, *args, platform=None, **kwargs):
        observed_platforms.append(platform)
        return task_run(self, *args, platform=platform, **kwargs)

    monkeypatch.setattr(Task, "run", observe_platform)

    executor.run_protocol(flipping, action, mode=ExecutionMode.FIT)

    assert len(observed_platforms) == 1
    fit_platform = observed_platforms[0]
    assert fit_platform is not None
    # The executor must pass a *reconstructed* platform, not the live one.
    assert fit_platform is not executor.platform
    # The reconstructed platform handed to Task.run must be a complete, usable
    # snapshot: same parameters and calibration, and no live hardware attached.
    assert fit_platform.parameters == executor.platform.parameters
    assert fit_platform.calibration == executor.platform.calibration
    assert fit_platform.instruments == {}
    assert not fit_platform.is_connected


SCRIPTS = Path(__file__).parent / "scripts"
CALIBRATION_SCRIPTS = Path(__file__).parent / "calibration_scripts"


@dataclass
class FakeParameters(Parameters):
    par: int


@dataclass
class FakeData(Data):
    par: int


@dataclass
class FakeResults(Results):
    par: dict[QubitId, int]


def _acquisition(params: FakeParameters, platform) -> FakeData:
    return FakeData(par=params.par)


def _fit(data: FakeData) -> FakeResults:
    return FakeResults(par={0: data.par})


def _plot(data: FakeData, target: QubitId, fit: FakeResults | None = None):
    pass


def _update(results: FakeResults, platform, qubit):
    pass


def test_calibration_task_uses_bound_executor(tmp_path, platform, mocker):
    acquire = mocker.spy(Executor, "acquire")
    fit = mocker.spy(Executor, "fit")
    protocol = Protocol(_acquisition, _fit, _plot, _update)
    task = Task(Action("fake", "fake", parameters={"par": 7}), protocol)

    completed = task.run(
        mode=ExecutionMode.ACQUIRE | ExecutionMode.FIT,
        folder=tmp_path,
        platform=platform,
        targets=[0],
    )

    acquire.assert_called_once()
    fit.assert_called_once()
    assert completed.data.par == 7
    assert completed.results.par == {0: 7}
    assert completed.task.targets == [0]
    assert completed.data_time >= 0
    assert completed.results_time >= 0


def test_calibration_without_optional_phases(tmp_path, platform):
    executor = Executor.create(tmp_path, targets=[0], platform=platform)
    executor.init(force=True)
    protocol = Protocol(_acquisition)
    action = Action("fake", "fake", parameters={"par": 7})

    completed = executor.run_protocol(protocol, action)
    assert completed.data.par == 7
    assert completed.results is None

    output = Output(executor.history, executor.meta, executor.platform)
    output.process(tmp_path, mode=ExecutionMode.FIT)
    assert next(output.history.values()).results is None
    executor.close()


@pytest.fixture
def fake_protocols(request):
    marker = request.node.get_closest_marker("protocols")
    if marker is None:
        return

    protocols = {}
    for name in marker.args:
        routine = Protocol(_acquisition, _fit, _plot, _update)
        setattr(qibocal.protocols, name, routine)
        protocols[name] = routine

    return protocols


@pytest.fixture
def executor(tmp_path: Path, platform: CalibrationPlatform):
    return Executor.create(tmp_path / "out", targets=[0])


def test_init(executor: Executor):
    init = executor.init

    init()
    with pytest.raises(RuntimeError, match="Directory .* already exists"):
        init()

    init(force=True)

    assert executor.meta is not None
    assert executor.meta.start is not None


def test_close(executor: Executor):
    executor.init()
    executor.close()

    assert executor.meta is not None
    assert executor.meta.start is not None
    assert executor.meta.end is not None


def test_context_manager(executor: Executor):
    executor.init()

    with executor:
        assert executor.meta is not None
        assert executor.meta.start is not None


def test_open(tmp_path: Path, platform: CalibrationPlatform):
    path = tmp_path / "my-open-folder"

    with Executor.open(path, targets=[0]) as e:
        assert isinstance(e.t1, Callable)
        assert e.meta is not None
        assert e.meta.start is not None

    assert e.meta.end is not None


def test_single_shot(tmp_path: Path, platform: CalibrationPlatform):
    globals_ = {"platform": platform, "targets": [0], "path": tmp_path}
    exec((CALIBRATION_SCRIPTS / "single_shot.py").read_text(), globals_)  # noqa: S102


def test_rx_calibration(tmp_path: Path, platform: CalibrationPlatform):
    globals_ = {"platform": platform, "target": 0, "path": tmp_path}
    exec((CALIBRATION_SCRIPTS / "rx_calibration.py").read_text(), globals_)  # noqa: S102


def test_check_input_qubit_overlap():
    """Verify input qubit overlap validation for single qubits and qubit pairs."""

    # list of unrepeated qubit, check passes
    inputs = ["B0", 1, "B2", 3]
    check_overlap_in_input_qubits(inputs)

    # list of repeated qubits, check raises a ValuError
    inputs = ["B0", 1, "B0", 3]
    with pytest.raises(ValueError, match="One or more target qubits were repeated."):
        check_overlap_in_input_qubits(inputs)

    # list of unrepeated qubit pairs and unrepeated qubits, check passes
    inputs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    check_overlap_in_input_qubits(inputs)

    # list of repeated pairs, check raises ValueError
    inputs = [(0, 1), (2, 3), (0, 1)]
    with pytest.raises(ValueError, match="One or more target qubits were repeated."):
        check_overlap_in_input_qubits(inputs)

    # list of unrepeated pairs but repeated qubit, check raises ValueError
    inputs = [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="One or more target qubits were repeated."):
        check_overlap_in_input_qubits(inputs)

    # list of unrepeated pair but same qubit in one pair, check raises ValueError
    inputs = [(0, 1), (2, 2)]
    with pytest.raises(ValueError, match="One or more target qubits were repeated."):
        check_overlap_in_input_qubits(inputs)
