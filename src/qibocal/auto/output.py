import getpass
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qibolab import Platform
from qibolab.serialize import dump_platform

from qibocal.auto.task import TaskId
from qibocal.config import log
from qibocal.version import __version__

from .history import History


@dataclass
class Versions:
    """Versions of the main software used."""

    other: dict
    qibocal: str = __version__


@dataclass
class TaskStats:
    acquisition: float
    fit: float

    @property
    def tot(self) -> float:
        return self.acquisition + self.fit


@dataclass
class Metadata:
    title: str
    backend: str
    platform: str
    start_time: datetime
    end_time: Optional[datetime]
    stats: dict[TaskId, TaskStats]
    versions: Versions
    more: Optional[dict] = None

    @classmethod
    def generate(cls, name: str, backend, platform: str):
        """Methods that takes care of:
        - dumping original platform
        - storing qq runcard
        - generating meta.yml
        """
        now = datetime.now(timezone.utc)
        versions = Versions(other=backend.versions)
        return cls(
            title=name,
            backend=backend.name,
            platform=platform,
            start_time=now,
            end_time=None,
            stats={},
            versions=versions,
        )

    def end(self):
        """Register completion time."""
        self.end_time = datetime.now(timezone.utc)


def _new_output() -> Path:
    user = getpass.getuser().replace(".", "-")
    date = datetime.now().strftime("%Y-%m-%d")

    num = 0
    while True:
        path = Path.cwd() / f"{date}-{str(num).rjust(3, '0')}-{user}"
        log.info(f"Trying to create directory {path}")

        if not path.exists():
            break

        log.info(f"Directory {path} already exists.")
        num += 1

    return path


RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform"
PLATFORM = "platform"
META = "meta.json"


@dataclass
class Output:
    """Output manager.

    This object represents the output folder, serializing from and
    deserialing to it.
    """

    history: History
    meta: Metadata
    platform: Optional[Platform] = None

    @classmethod
    def load(cls, path: Path):
        return cls(
            history=History.load(path),
            meta=Metadata(**json.loads((path / META).read_text())),
        )

    @staticmethod
    def mkdir(path: Optional[Path] = None, force: bool = False):
        if path is None:
            path = _new_output()
        elif path.exists() and not force:
            raise RuntimeError(f"Directory {path} already exists.")
        elif path.exists() and force:
            log.warning(f"Deleting previous directory {path}.")
            shutil.rmtree(path)

        log.info(f"Creating directory {path}.")
        path.mkdir(parents=True)
        return path

    def dump(self, path: Path):
        # dump metadata
        self._export_stats()
        (path / META).write_text(json.dumps(asdict(self.meta), indent=4))

        # dump tasks
        self.history.flush(path)

        # dump platform used
        # if the original one is not defined, use the current one as the original,
        # else update the new one
        if self.platform is not None:
            platpath = path / PLATFORM
            if platpath.is_dir():
                platpath = path / UPDATED_PLATFORM

            platpath.mkdir(parents=True, exist_ok=True)
            dump_platform(self.platform, platpath)

    def _export_stats(self):
        """Export task statistics.

        Extract statistics from the history, and record them in the
        metadata.
        """
        self.meta.stats = {
            id: TaskStats(completed.data_time, completed.results_time)
            for id, completed in self.history.items()
        }
