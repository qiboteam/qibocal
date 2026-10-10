import json
from copy import deepcopy
from pathlib import Path

import pytest
from qibo.backends import construct_backend

from qibocal.auto.mode import ExecutionMode
from qibocal.auto.notes import NOTES, Note, load_notes
from qibocal.auto.output import History, Metadata, Output, TaskStats, _new_output
from qibocal.auto.runcard import Action, Runcard
from qibocal.calibration.platform import CalibrationPlatform
from qibocal.cli.fit import fit

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


# TODO: this is essentially a proto `qq run` invocation, it should be simplified as
# much as possible in the library, and made available in conftest
@pytest.fixture
def mock_output(tmp_path: Path, platform: CalibrationPlatform) -> tuple[Output, Path]:
    backend = construct_backend(backend="qibolab", platform=platform)
    meta = Metadata.generate(backend)
    output = Output(History(), meta, platform)
    platform.connect()
    meta.start()
    output.history = Runcard(
        actions=[deepcopy(ACTION)], targets=list(platform.qubits)
    ).run(output=tmp_path, platform=platform, mode=ExecutionMode.ACQUIRE)
    meta.end()
    platform.disconnect()
    output.dump(tmp_path)

    return output, tmp_path


def test_output_process(mock_output: tuple[Output, Path]):
    """Create method of Executor."""
    output, path = mock_output
    # perform fit
    output.process(path, mode=ExecutionMode.FIT)

    # check double fit error
    with pytest.raises(KeyError):
        output.process(path, mode=ExecutionMode.FIT)


def test_task_stats():
    stats = TaskStats(2, 5)
    assert stats.fit == 5
    assert stats.tot == 7


def test_output_notes(mock_output):
    output, path = mock_output
    task_id = next(iter(output.history))
    completed = output.history[task_id]
    assert output.notes == completed.notes == []
    assert json.loads((path / NOTES).read_text()) == []
    assert json.loads((completed.path / NOTES).read_text()) == []

    session_note = Note(content="Session finding")
    protocol_notes = [
        Note(content="Initial finding", author="agent"),
        Note(content="Follow-up finding", author="agent"),
    ]
    output.notes.append(session_note)
    completed.notes.extend(protocol_notes)
    output.dump(path)

    loaded = Output.load(path)
    assert loaded.notes == [session_note]
    assert loaded.history[task_id].notes == protocol_notes
    loaded.process(path, mode=ExecutionMode.FIT)
    loaded.dump(path)
    loaded.process(path, mode=ExecutionMode.FIT, force=True)
    loaded.dump(path)
    assert Output.load(path).notes == [session_note]
    assert Output.load(path).history[task_id].notes == protocol_notes


def test_output_notes_only(mock_output):
    output, path = mock_output
    task_id = next(iter(output.history))
    completed = output.history[task_id]
    meta = (path / "meta.json").read_bytes()
    data = (completed.path / "data.npz").read_bytes()
    output.notes.append(Note(content="Session finding"))
    completed.notes.append(Note(content="Protocol finding"))
    output.dump_notes(path)
    completed.dump_notes()
    assert load_notes(path) == output.notes
    assert load_notes(completed.path) == completed.notes
    assert (path / "meta.json").read_bytes() == meta
    assert (completed.path / "data.npz").read_bytes() == data


def test_output_legacy_notes(mock_output):
    output, path = mock_output
    task_id = next(iter(output.history))
    (path / NOTES).unlink()
    (output.history[task_id].path / NOTES).unlink()
    loaded = Output.load(path)
    assert loaded.notes == []
    assert loaded.history[task_id].notes == []


def test_notes_fit_copy(mock_output, tmp_path):
    output, path = mock_output
    task_id = next(iter(output.history))
    output.notes.append(Note(content="Session finding"))
    output.history[task_id].notes.append(Note(content="Protocol finding"))
    output.dump(path)
    copied = tmp_path / "copied"
    fit(path, update=False, output_path=copied, force=False)
    loaded = Output.load(copied)
    assert loaded.notes == output.notes
    assert loaded.history[task_id].notes == output.history[task_id].notes
    assert Output.load(path).history[task_id].results is None


def test_notes_iteration_isolation(mock_output):
    output, path = mock_output
    first_id = next(iter(output.history))
    first = output.history[first_id]
    output.platform.connect()
    second = first.task.run(
        mode=ExecutionMode.ACQUIRE,
        folder=path / "data" / f"{first_id.id}-1",
        platform=output.platform,
    )
    output.platform.disconnect()
    second_id = output.history.push(second)
    first.notes.append(Note(content="First iteration"))
    second.notes.append(Note(content="Second iteration"))
    output.dump(path)
    loaded = Output.load(path)
    assert loaded.history[first_id].notes == first.notes
    assert loaded.history[second_id].notes == second.notes
    assert first.notes != second.notes


def test_new_output():
    path1 = _new_output()
    path1.mkdir()
    path2 = _new_output()
    assert path1.name.split("-")[3] == "000"
    assert path2.name.split("-")[3] == "001"


def test_output_mkdir():
    path1 = Output.mkdir()
    path2 = Output.mkdir()
    assert path1.name.split("-")[3] == "000"
    assert path2.name.split("-")[3] == "001"

    with pytest.raises(RuntimeError):
        Output.mkdir(path1)

    Output.mkdir(path1, force=True)
