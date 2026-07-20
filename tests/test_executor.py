from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from qibolab import Platform

import qibocal
import qibocal.protocols
from qibocal import Executor
from qibocal.auto.execute import check_overlap_in_input_qubits
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.auto.runcard import Action
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
    exec((CALIBRATION_SCRIPTS / "single_shot.py").read_text(), globals_)


def test_rx_calibration(tmp_path: Path, platform: CalibrationPlatform):
    globals_ = {"platform": platform, "target": 0, "path": tmp_path}
    exec((CALIBRATION_SCRIPTS / "rx_calibration.py").read_text(), globals_)


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
