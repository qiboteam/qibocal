import getpass
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qibo.backends import construct_backend
from qibolab import Platform
from qibolab.serialize import dump_platform

from ..config import log
from ..version import __version__
from .history import History
from .mode import ExecutionMode
from .task import Targets


@dataclass
class Versions:
    """Versions of the main software used."""

    other: dict
    qibocal: str = __version__


@dataclass
class TaskStats:
    """Statistics about task execution."""

    acquisition: float
    """Acquisition timing."""
    fit: float
    """Fitting timing."""

    @property
    def tot(self) -> float:
        """Total execution time."""
        return self.acquisition + self.fit


@dataclass
class Metadata:
    """Execution metadata."""

    title: str
    backend: str
    platform: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    stats: dict[str, TaskStats]
    versions: Versions
    author: Optional[str] = None
    tag: Optional[str] = None
    targets: Optional[Targets] = None

    @classmethod
    def generate(cls, name: str, backend):
        """Generate template metadata.

        The purpose is to fill the arguments with defaults, or extract
        them from the few arguments.
        """
        versions = Versions(other=backend.versions)
        return cls(
            title=name,
            backend=backend.name,
            platform=backend.platform.name,
            start_time=None,
            end_time=None,
            stats={},
            versions=versions,
        )

    def start(self):
        """Register start time."""
        self.start_time = datetime.now(timezone.utc)

    def end(self):
        """Register completion time."""
        self.end_time = datetime.now(timezone.utc)

    @classmethod
    def load(cls, path):
        attrs: dict = json.loads((path / META).read_text())
        del attrs["date"]
        del attrs["start-time"]
        del attrs["end-time"]
        attrs["start_time"] = (
            datetime.fromisoformat(attrs["start_time"])
            if attrs["start_time"] is not None
            else None
        )
        attrs["end_time"] = (
            datetime.fromisoformat(attrs["end_time"])
            if attrs["end_time"] is not None
            else None
        )
        versions = attrs.pop("versions")
        attrs["versions"] = Versions(qibocal=versions.pop("qibocal"), other=versions)
        return cls(**attrs)

    def dump(self) -> dict:
        """Dump to serializable to dictionary."""
        d = asdict(self)
        d["start_time"] = str(d["start_time"]) if self.start_time is not None else None
        d["end_time"] = str(d["end_time"]) if self.end_time is not None else None
        versions = d.pop("versions")
        d["versions"] = versions["other"] | {"qibocal": versions["qibocal"]}
        d["date"] = str(self.start_time.date()) if self.start_time is not None else None
        d["start-time"] = (
            self.start_time.strftime("%H:%M:%S")
            if self.start_time is not None
            else None
        )
        d["end-time"] = (
            self.end_time.strftime("%H:%M:%S") if self.end_time is not None else None
        )

        return d


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
        """Load output from existing folder."""
        return cls(
            history=History.load(path),
            meta=Metadata.load(path),
        )

    @staticmethod
    def mkdir(path: Optional[Path] = None, force: bool = False) -> Path:
        """Create output directory.

        If a `path` is given and existing, it is overwritten only in the case `force`
        is enabled, otherwise an error is thrown. If not already existing, it is just
        used.

        If no `path` is given, a default one is generated (according to user name and
        time stamp).
        """
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
        """Dump output content to an output folder."""
        # dump metadata
        self._export_stats()
        (path / META).write_text(json.dumps(self.meta.dump(), indent=4))

        # dump tasks
        self.history.flush(path)

        # update platform
        if self.platform is not None:
            self.update_platform(self.platform, path)

    @staticmethod
    def update_platform(platform: Platform, path: Path):
        """Dump platform used.

        If the original one is not defined, use the current one as the
        original, else update the new one.
        """
        platpath = path / PLATFORM
        if platpath.is_dir():
            platpath = path / UPDATED_PLATFORM

        platpath.mkdir(parents=True, exist_ok=True)
        dump_platform(platform, platpath)

    def _export_stats(self):
        """Export task statistics.

        Extract statistics from the history, and record them in the
        metadata.
        """
        self.meta.stats = {
            str(id): TaskStats(completed.data_time, completed.results_time)
            for id, completed in self.history.items()
        }

    def process(
        self,
        output: Path,
        mode: ExecutionMode,
        update: bool = True,
        force: bool = False,
    ):
        """Process existing output."""
        self.platform = construct_backend(
            backend=self.meta.backend, platform=self.meta.platform
        ).platform
        assert self.platform is not None

        for task_id, completed in self.history.items():
            # TODO: should we drop this check as well, and just allow overwriting?
            if (
                ExecutionMode.FIT in mode
                and not force
                and completed.results is not None
            ):
                raise KeyError(f"{task_id} already contains fitting results.")

            # TODO: this is a plain hack, to be fixed together with the task lifecycle
            # refactor
            self.history._tasks[task_id.id][task_id.iteration] = completed.task.run(
                platform=self.platform, mode=mode, folder=completed.path
            )
            self.history.flush(output)

            if update and completed.task.update:
                completed.update_platform(platform=self.platform)
