import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from qibocal import Executor, Note, Protocol
from qibocal.auto.notes import NOTES, dump_notes, load_notes
from qibocal.auto.task import Completed


def test_note_defaults():
    before = datetime.now(UTC)
    note = Note(content=" A finding. ")
    assert note.content == "A finding."
    assert before <= note.timestamp <= datetime.now(UTC)
    assert note.timestamp.tzinfo is UTC
    assert note.author is None
    with pytest.raises(ValidationError, match="frozen"):
        note.content = "Changed"


def test_note_timestamp():
    timestamp = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    note = Note(content="Finding", timestamp=timestamp, author="agent")
    assert note.timestamp == datetime(2026, 1, 1, 10, tzinfo=UTC)
    assert note.timestamp.tzinfo is UTC
    assert note.author == "agent"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": ""},
        {"content": " \n "},
        {"content": "Finding", "timestamp": "2026-01-01T00:00:00"},
        {"content": "Finding", "timestamp": "invalid"},
        {"content": "Finding", "author": 1},
    ],
)
def test_invalid_note(kwargs):
    with pytest.raises(ValidationError):
        Note(**kwargs)


def test_notes_roundtrip(tmp_path):
    notes = [
        Note(content="First finding", author="agent"),
        Note(content="Updated finding"),
    ]
    dump_notes(notes, tmp_path)
    assert load_notes(tmp_path) == notes
    raw = json.loads((tmp_path / NOTES).read_text())
    assert len(raw) == 2
    assert raw[0] == {
        "content": notes[0].content,
        "timestamp": notes[0].timestamp.isoformat().replace("+00:00", "Z"),
        "author": "agent",
    }
    assert raw[1]["author"] is None


def test_legacy_notes(tmp_path):
    assert load_notes(tmp_path) == []
    dump_notes([], tmp_path)
    assert json.loads((tmp_path / NOTES).read_text()) == []


@pytest.mark.parametrize("raw", ["not json", "{}", '[{"content": ""}]'])
def test_invalid_stored_notes(tmp_path, raw):
    (tmp_path / NOTES).write_text(raw)
    with pytest.raises(ValidationError):
        load_notes(tmp_path)


def test_protocol_notes(tmp_path):
    completed = Completed(path=tmp_path / "protocol")
    completed.notes.append(Note(content="Finding"))
    completed.dump_notes()
    assert load_notes(completed.path) == completed.notes
    assert Completed().notes == []
    with pytest.raises(ValueError, match="output path"):
        Completed().dump_notes()


def test_direct_fit_preserves_notes():
    def acquire(samples: int) -> int:
        return samples

    def fit(data: int) -> int:
        return data + 1

    executor = Executor(None, targets=[])
    completed = executor.acquire(Protocol(acquire, fit=fit)(pars=3))
    completed.notes.append(Note(content="Before fit"))
    fitted = executor.fit(completed)
    assert fitted.notes == completed.notes
    assert fitted.results == 4


def test_executor_session_notes(tmp_path, platform):
    path = tmp_path / "session"
    note = Note(content="Session finding")
    with Executor.open(path, platform=platform) as executor:
        assert executor.notes == []
        executor.notes.append(note)
    assert load_notes(path) == [note]
    with executor:
        executor.notes.append(Note(content="Follow-up finding"))
    assert load_notes(path) == executor.notes
