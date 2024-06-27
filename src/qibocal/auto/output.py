import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .history import History


@dataclass
class Versions:
    qibocal: str
    other: dict


META_FILE = "meta.json"


@dataclass
class Metadata:
    title: str
    backend: str
    platform: str
    start_time: datetime
    end_time: Optional[datetime]
    versions: Versions
    more: Optional[dict] = None

    @classmethod
    def generate(cls, backend, platform, path: Path):
        """Methods that takes care of:
        - dumping original platform
        - storing qq runcard
        - generating meta.yml
        """

        import qibocal

        now = datetime.now(timezone.utc)
        versions = Versions(qibocal=qibocal.__version__, other=backend.versions)
        return cls(
            title=path.name,
            backend=backend.name,
            platform=str(platform),
            start_time=now,
            end_time=None,
            versions=versions,
        )


@dataclass
class Output:
    history: History
    meta: Metadata

    @classmethod
    def load(cls, path: Path):
        return cls(
            history=History.load(path),
            meta=Metadata(**json.loads((path / META_FILE).read_text())),
        )
