"""Tasks execution."""

from dataclasses import dataclass
from typing import Union

from qibolab import create_platform
from qibolab.platform import Platform
from qibolab.serialize import dump_platform

from qibocal.config import log

from .history import History
from .mode import ExecutionMode
from .operation import Routine
from .task import Action, Completed, Targets, Task

PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    history: History
    """The execution history, with results and exit states."""
    targets: Targets
    """Qubits/Qubit Pairs to be calibrated."""
    platform: Platform
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""

    @classmethod
    def create(cls, platform: Union[Platform, str]):
        """Load list of protocols."""
        platform = (
            platform if isinstance(platform, Platform) else create_platform(platform)
        )
        return cls(history=History(), platform=platform, targets=list(platform.qubits))

    def run_protocol(
        self,
        protocol: Routine,
        parameters: Union[dict, Action],
        mode: ExecutionMode = ExecutionMode.ACQUIRE | ExecutionMode.FIT,
        update: bool = True,
    ) -> Completed:
        """Run single protocol in ExecutionMode mode."""
        action = Action.cast(source=parameters, operation=str(protocol))
        task = Task(action=action, operation=protocol)
        log.info(f"Executing mode {mode} on {task.action.id}.")

        completed = task.run(platform=self.platform, targets=self.targets, mode=mode)
        self.history.push(completed)

        # TODO: drop, as the conditions won't be necessary any longer, and then it could
        # be performed as part of `task.run` https://github.com/qiboteam/qibocal/issues/910
        if ExecutionMode.FIT in mode:
            if self.update and task.update:
                completed.update_platform(platform=self.platform)

        return completed
