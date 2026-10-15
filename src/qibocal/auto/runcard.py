"""Specify runcard layout, handles (de)serialization."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic.dataclasses import dataclass

from qibocal.calibration.platform import CalibrationPlatform
from qibocal.config import log

from .. import protocols
from .execute import check_overlap_in_input_qubits
from .history import History
from .mode import ExecutionMode
from .output import PLATFORM
from .task import Action, Targets, Task

RUNCARD = "runcard.yml"
"""Runcard filename."""


@dataclass
class Runcard:
    """Structure of an execution runcard."""

    actions: list[Action]
    """List of action to be executed."""
    targets: Targets | None = None
    """Qubits to be calibrated.

    If `None` the protocols will be executed on all qubits
    available in the platform.
    """
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = "mock"
    """Qibolab platform."""
    update: bool = True

    @classmethod
    def load(cls, runcard: dict[str, Any] | Path):
        """Load a runcard dict or path."""
        if not isinstance(runcard, dict):
            return cls(yaml.safe_load((runcard / RUNCARD).read_text(encoding="utf-8")))
        return cls(**runcard)

    def dump(self, path):
        """Dump runcard object to yaml."""
        (path / RUNCARD).write_text(yaml.safe_dump(asdict(self)), encoding="utf-8")

    def run(
        self,
        output: Path,
        platform: CalibrationPlatform,
        mode: ExecutionMode,
        update: bool = True,
    ) -> History:
        """Run runcard and dump to output."""
        targets = self.targets if self.targets is not None else list(platform.qubits)
        check_overlap_in_input_qubits(targets)
        history = History.load(output)
        update = update and self.update
        for action in self.actions:
            protocol = getattr(protocols, action.operation)
            task = Task(action=action, operation=protocol)
            log.info(f"Executing mode {mode} on {task.action.id}.")
            completed = task.run(
                platform=(
                    platform
                    if ExecutionMode.ACQUIRE in mode
                    else CalibrationPlatform.from_datafolder(
                        folder_path=output / PLATFORM,
                        platform_name=platform.name,
                    )
                ),
                targets=targets,
                mode=mode,
                folder=history.task_path(history._pending_task_id(task.id), output),
            )
            history.push(completed)
            if (
                ExecutionMode.FIT in mode
                and update
                and task.update
                and protocol.update is not None
                and completed.results is not None
            ):
                completed.update_platform(platform=platform)
        history.dump(output)
        return history
