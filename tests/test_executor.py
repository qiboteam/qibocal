from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2

import numpy as np
import pytest
import yaml
from qibo import Circuit
from qibolab import Platform

import qibocal
import qibocal.protocols
from qibocal import Executor
from qibocal.auto.execute import _action_parameters, check_overlap_in_input_qubits
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitId,
    Results,
)
from qibocal.auto.output import PLATFORM, Output
from qibocal.auto.runcard import Action, Runcard
from qibocal.auto.task import SINGLE_ACTION, Completed, Task
from qibocal.calibration.platform import (
    CalibrationPlatform,
    create_calibration_platform,
)
from qibocal.cli.report import report
from qibocal.protocols.utils import PowerLevel

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
    def acquire(parameters):
        pass

    def fit(data, fitpars):
        pass

    def report(data, results, reportpars):
        pass

    def update(results, platform):
        pass

    protocol = Protocol(
        acquisition=mocker.create_autospec(acquire, return_value=7),
        fit=mocker.create_autospec(fit, return_value=11),
        report=mocker.create_autospec(report, return_value="report"),
        update=mocker.create_autospec(update),
    )
    return protocol(pars=3, fit={"fit": True}, report={"report": True})


def test_bound_protocol_executor(bound_protocol, platform):
    executor = Executor(platform, targets=[])
    completed = executor(bound_protocol)

    assert isinstance(completed, Completed)
    assert completed.data == 7
    assert completed.results == 11
    assert completed.reports == "report"
    bound_protocol.protocol.acquisition.assert_called_once_with(3)
    bound_protocol.protocol.fit.assert_called_once_with(7, fitpars={"fit": True})
    bound_protocol.protocol.report.assert_called_once_with(
        7, results=11, reportpars={"report": True}
    )
    bound_protocol.protocol.update.assert_called_once_with(11, platform=platform)
    assert executor.path is None
    assert list(executor.history) == []


@pytest.mark.parametrize("skip_fit", [False, True])
def test_bound_protocol_without_fit(bound_protocol, platform, skip_fit):
    fit = bound_protocol.protocol.fit
    if not skip_fit:
        bound_protocol.protocol.fit = None
    completed = Executor(platform, targets=[])(bound_protocol, skip_fit=skip_fit)

    assert completed.data == 7
    assert completed.results is None
    assert completed.reports is None
    fit.assert_not_called()
    bound_protocol.protocol.report.assert_not_called()
    bound_protocol.protocol.update.assert_not_called()


def test_bound_protocol_optional_report_and_update(bound_protocol, platform):
    bound_protocol.protocol.report = None
    bound_protocol.protocol.update = None
    completed = Executor(platform, targets=[])(bound_protocol)

    assert completed.results == 11


def test_bound_protocol_update_disabled(bound_protocol, platform):
    executor = Executor(platform, targets=[], update=False)
    completed = executor(bound_protocol)

    bound_protocol.protocol.update.assert_not_called()
    executor.update(completed)
    bound_protocol.protocol.update.assert_called_once_with(11, platform=platform)


def test_bound_protocol_individual_phases(bound_protocol, platform):
    executor = Executor(platform, targets=[])
    data = executor.acquire(bound_protocol)
    results = executor.fit(data)
    assert executor.report(results).reports == "report"
    executor.update(results)

    assert data is not results
    assert data.results is None
    assert results.data == data.data
    assert results.bound is bound_protocol

    bound_protocol.protocol.acquisition.assert_called_once_with(3)
    bound_protocol.protocol.fit.assert_called_once_with(7, fitpars={"fit": True})
    bound_protocol.protocol.report.assert_called_once_with(
        7, results=11, reportpars={"report": True}
    )
    bound_protocol.protocol.update.assert_called_once_with(11, platform=platform)


@pytest.mark.parametrize("phase", ["fit", "report", "update"])
def test_bound_protocol_missing_phase(bound_protocol, platform, phase):
    setattr(bound_protocol.protocol, phase, None)
    executor = Executor(platform, targets=[])
    completed = executor.acquire(bound_protocol)

    with pytest.raises(ValueError, match="Protocol does not support"):
        getattr(executor, phase)(completed)


def test_bound_protocol_update_requires_platform(bound_protocol):
    with pytest.raises(ValueError, match="does not have a platform"):
        Executor(None).update(
            Completed(bound=bound_protocol, _data=7, _results=11, _targets=[])
        )


@pytest.mark.parametrize("phase", ["acquisition", "fit", "report", "update"])
def test_bound_protocol_propagates_errors(bound_protocol, platform, phase):
    getattr(bound_protocol.protocol, phase).side_effect = RuntimeError(phase)

    with pytest.raises(RuntimeError, match=phase):
        Executor(platform, targets=[])(bound_protocol)


@pytest.mark.parametrize("params", [ACTION, PARAMETERS])
def test_runcard_acquisition(
    params: dict | Action, platform: Platform | str, tmp_path: Path
):
    platform = (
        platform
        if isinstance(platform, Platform)
        else create_calibration_platform(platform)
    )
    runcard = Runcard(
        actions=[deepcopy(Action.cast(params, "flipping"))],
        targets=list(platform.qubits),
    )
    history = runcard.run(
        platform=platform,
        output=tmp_path,
        mode=ExecutionMode.ACQUIRE,
    )
    assert next(history.values()).data is not None


def test_runcard_fit_reconstructs_platform_from_datafolder(
    executor: Executor, monkeypatch
):
    """Verifies that when a runcard runs a FIT-only protocol, it reconstructs
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
    runcard = Runcard(actions=[action], targets=executor.targets)
    history = runcard.run(
        output=executor.path, platform=executor.platform, mode=ExecutionMode.ACQUIRE
    )

    acquired_folder = history.task_path(
        history._executed_task_id(action.id), executor.path
    )
    fit_folder = history.task_path(history._pending_task_id(action.id), executor.path)
    fit_folder.mkdir(parents=True)
    # FIT creates the next task iteration, and Task.run loads data from that folder.
    # Copy the acquisition payload there so the FIT call can run without reacquiring.
    for data_file in acquired_folder.glob("data.*"):
        copy2(data_file, fit_folder)

    observed_platforms = []
    # Save the original before patching: the wrapper records the runcard's platform choice,
    # then delegates to Task.run so the normal data loading and fit still happen.
    task_run = Task.run

    def observe_platform(self, *args, platform=None, **kwargs):
        observed_platforms.append(platform)
        return task_run(self, *args, platform=platform, **kwargs)

    monkeypatch.setattr(Task, "run", observe_platform)

    runcard.run(
        output=executor.path, platform=executor.platform, mode=ExecutionMode.FIT
    )

    assert len(observed_platforms) == 1
    fit_platform = observed_platforms[0]
    assert fit_platform is not None
    # The runcard must pass a *reconstructed* platform, not the live one.
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


def test_completed_load_and_refit(tmp_path, platform, monkeypatch):
    protocol = Protocol(_acquisition, _fit, _plot, _update)
    monkeypatch.setattr(qibocal.protocols, "fake", protocol, raising=False)
    task = Task(Action("fake", "fake", targets=[0], parameters={"par": 7}), protocol)
    original = task.run(
        mode=ExecutionMode.ACQUIRE | ExecutionMode.FIT,
        folder=tmp_path,
        platform=platform,
    )
    loaded = Completed.load(tmp_path)

    assert loaded.bound.protocol == protocol
    assert loaded.bound.parameters.par == 7
    assert loaded.data.par == original.data.par
    assert loaded.results.par == original.results.par
    assert loaded.targets == [0]

    refitted = Executor(None).fit(loaded)
    assert refitted is not loaded
    assert refitted.bound is loaded.bound
    assert refitted.task == loaded.task
    assert refitted.task is not loaded.task
    assert refitted.path == tmp_path
    assert refitted.data is loaded.data
    assert refitted.results is not loaded.results
    assert refitted.results.par == loaded.results.par


def test_calibration_without_optional_phases(tmp_path, platform, monkeypatch):
    executor = Executor.create(tmp_path, targets=[0], platform=platform)
    executor._init(force=True)
    protocol = Protocol(_acquisition)
    monkeypatch.setattr(qibocal.protocols, "fake", protocol, raising=False)
    action = Action("fake", "fake", parameters={"par": 7})

    executor.history = Runcard(actions=[action], targets=[0]).run(
        output=tmp_path,
        platform=executor.platform,
        mode=ExecutionMode.ACQUIRE | ExecutionMode.FIT,
    )
    completed = next(executor.history.values())
    assert completed.data.par == 7
    assert completed.results is None

    output = Output(executor.history, executor.meta, executor.platform)
    output.process(tmp_path, mode=ExecutionMode.FIT)
    assert next(output.history.values()).results is None
    executor.close()


@pytest.fixture
def executor(tmp_path: Path, platform: CalibrationPlatform):
    return Executor.create(tmp_path / "out", targets=[0])


def test_init(executor: Executor, mocker):
    mkdir = mocker.spy(Output, "mkdir")
    dump = mocker.spy(Output, "dump")
    start = mocker.spy(executor.meta, "start")
    connect = mocker.spy(executor.platform, "connect")

    assert not executor._initialized
    executor._init()
    marker = executor.path / "keep"
    marker.touch()
    executor._init()
    executor._init(force=True)

    mkdir.assert_called_once_with(executor.path, False)
    dump.assert_called_once()
    start.assert_called_once_with()
    assert connect.call_count == 3
    assert executor._initialized
    assert executor.meta.start_time is not None
    assert marker.exists()


def test_init_existing_directory(executor: Executor):
    executor.path.mkdir()

    with pytest.raises(RuntimeError, match="Directory .* already exists"):
        executor._init()

    assert not executor._initialized
    assert not executor.platform.is_connected
    assert executor.meta.start_time is None

    executor._init(force=True)
    assert executor._initialized
    assert executor.platform.is_connected


def test_init_connection_failure(executor: Executor, mocker):
    connect = mocker.patch.object(
        executor.platform, "connect", side_effect=RuntimeError("Connection failed")
    )
    executor_init = mocker.spy(Output, "dump")

    with pytest.raises(RuntimeError, match="Connection failed"):
        executor._init()

    assert executor._initialized
    assert executor.meta.start_time is not None
    connect.side_effect = None
    executor._init()
    executor_init.assert_called_once()
    assert connect.call_count == 2


def test_close(executor: Executor):
    executor._init()
    executor.close()

    assert executor.meta is not None
    assert executor.meta.start is not None
    assert executor.meta.end is not None


def test_context_manager(executor: Executor, mocker):
    init = mocker.spy(executor, "_init")
    connect = mocker.spy(executor.platform, "connect")
    disconnect = mocker.spy(executor.platform, "disconnect")

    with executor as entered:
        assert entered is executor
        assert executor.meta is not None
        assert executor.meta.start_time is not None
        assert executor.platform.is_connected
        start_time = executor.meta.start_time
        marker = executor.path / "keep"
        marker.touch()

    assert not executor.platform.is_connected
    with executor:
        assert executor.meta.start_time == start_time
        assert executor.platform.is_connected
        assert marker.exists()

    assert not executor.platform.is_connected
    assert init.call_count == 2
    assert connect.call_count == 2
    assert disconnect.call_count == 2


def test_open(tmp_path: Path, platform: CalibrationPlatform):
    path = tmp_path / "my-open-folder"

    with Executor.open(path, targets=[0]) as e:
        assert e.meta is not None
        assert e.meta.start is not None

    assert e.meta.end is not None


def test_open_records_executions(tmp_path, platform):
    path = tmp_path / "report"
    with Executor.open(path, platform=platform, targets=[0], force=True) as e:
        e(
            qibocal.protocols.standard_rb_2q(depths=[1, 2], niter=2, nshots=10),
            targets=[[0, 1]],
        )
        e(qibocal.protocols.rabi_amplitude(min_amp=0, max_amp=1, step_amp=0.1))
        e(qibocal.protocols.rabi_amplitude(min_amp=0, max_amp=0.5, step_amp=0.1))
        # recorded as soon as executed
        assert [str(task_id) for task_id in Output.load(path).history] == [
            "standard_rb_2q-0",
            "rabi_amplitude-0",
            "rabi_amplitude-1",
        ]

    report(e.path, e.history)

    assert (path / "index.html").is_file()
    loaded = Output.load(path).history["rabi_amplitude"][1]
    recorded = e.history["rabi_amplitude"][1]
    assert loaded.task.parameters.max_amp == 0.5
    assert loaded.task.targets == [0]
    assert loaded.results.amplitude == recorded.results.amplitude
    assert e.history["standard_rb_2q"][0].task.targets == [(0, 1)]


def test_open_records_normalized_parameters(tmp_path, platform):
    path = tmp_path / "report"
    frequency = ("lincenter", np.float64(1e6), np.int64(5))
    with Executor.open(path, platform=platform, targets=[0], force=True) as e:
        e(
            qibocal.protocols.resonator_spectroscopy(
                frequency=frequency, power_level="low", nshots=10
            )
        )

    action = yaml.safe_load(
        (path / "data" / "resonator_spectroscopy-0" / SINGLE_ACTION).read_text()
    )
    assert action["parameters"]["power_level"] == "low"
    assert action["parameters"]["frequency"] == ["lincenter", 1e6, 5]
    loaded = Output.load(path).history["resonator_spectroscopy"][0]
    assert loaded.task.parameters.power_level is PowerLevel.low


def test_action_parameters_plain_types():
    @dataclass
    class NormalizedParameters(Parameters):
        level: PowerLevel = PowerLevel.high
        values: tuple = ((np.float64(0.5), PowerLevel.low), np.int64(2))
        array: np.ndarray = field(default_factory=lambda: np.arange(3))
        mapping: dict = field(default_factory=lambda: {"step": np.float32(0.25)})
        circuit: Circuit = field(default_factory=lambda: Circuit(1))

    parameters = _action_parameters(NormalizedParameters())
    assert parameters["level"] == "high" and type(parameters["level"]) is str
    assert parameters["values"] == [[0.5, "low"], 2]
    assert parameters["array"] == [0, 1, 2]
    assert parameters["mapping"] == {"step": 0.25}
    # circuits are dumped to their own file by Action.dump
    assert isinstance(parameters["circuit"], Circuit)
    del parameters["circuit"]
    assert yaml.safe_load(yaml.safe_dump(parameters)) == parameters


def test_records_only_initialized_executions(tmp_path, platform, bound_protocol):
    bound = qibocal.protocols.rabi_amplitude(min_amp=0, max_amp=1, step_amp=0.1)
    executor = Executor.create(tmp_path / "out", platform=platform, targets=[0])
    assert executor(bound).task is None
    assert list(executor.history) == []
    assert not executor.path.exists()

    with executor:
        with pytest.warns(UserWarning, match="not registered"):
            executor(bound_protocol, targets=[])
        executor.fit(executor.acquire(bound))
        assert list(executor.history) == []


def test_open_without_default_targets(tmp_path, platform, bound_protocol):
    with Executor.open(tmp_path / "out", platform=platform) as executor:
        assert executor.targets is None
        assert executor(bound_protocol, targets=[]).data == 7
        with pytest.raises(ValueError, match="Targets must be supplied"):
            executor(bound_protocol)


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
