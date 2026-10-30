"""Persistent comments on calibration sessions and protocol executions."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

NOTES = "notes.json"
"""Comment history file, relative to a session or protocol output folder."""


class Note(BaseModel):
    """A comment with its creation time and optional externally supplied author."""

    model_config = ConfigDict(frozen=True)

    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    author: str | None = None

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Store all timestamps in UTC."""
        return value.astimezone(UTC)


_notes = TypeAdapter(list[Note])


def load_notes(path: Path) -> list[Note]:
    """Load comment history, treating outputs without notes as empty."""
    file = path / NOTES
    if not file.exists():
        return []
    return _notes.validate_json(file.read_text(encoding="utf-8"))


def dump_notes(notes: list[Note], path: Path):
    """Save comment history without touching calibration data or metadata."""
    validated = _notes.validate_python(notes)
    path.mkdir(parents=True, exist_ok=True)
    (path / NOTES).write_bytes(_notes.dump_json(validated, indent=4))
