"""Tasks execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from qibolab import create_platform
from qibolab.platform import Platform
from qibolab.serialize import dump_platform

from qibocal import protocols
from qibocal.config import log, raise_error

from .history import History
from .mode import ExecutionMode
from .operation import Routine
from .runcard import Action, Runcard, Targets
from .task import Completed, Task

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
        if isinstance(parameters, dict):
            parameters["operation"] = str(protocol)
            action = Action(**parameters)
        else:
            action = parameters
        task = Task(action, protocol)
        if isinstance(mode, ExecutionMode):
            log.info(f"Executing mode {mode} on {task.id}.")

        if ExecutionMode.ACQUIRE in mode and task.id in self.history:
            raise_error(KeyError, f"{task.id} already contains acquisition data.")
        if ExecutionMode.FIT is mode and self.history[task.id]._results is not None:
            raise_error(KeyError, f"{task.id} already contains fitting results.")

        completed = task.run(platform=self.platform, targets=self.targets, mode=mode)

        if ExecutionMode.FIT in mode and self.platform is not None:
            completed.update_platform(platform=self.platform, update=self.update)

        if self.platform is not None:
            (completed.datapath / PLATFORM_DIR).mkdir(parents=True, exist_ok=True)
            dump_platform(self.platform, completed.datapath / PLATFORM_DIR)

        self.history.push(completed)
        completed.dump()

        return completed


def run(runcard: Runcard, output: Path, mode: ExecutionMode, update: bool = True):
    """Run runcard and dump to output."""
    platform = runcard.platform_obj
    targets = runcard.targets if runcard.targets is not None else list(platform.qubits)
    instance = Executor(
        history=History.load(output),
        platform=platform,
        targets=targets,
        update=runcard.update,
    )

    for action in runcard.actions:
        instance.run_protocol(
            protocol=getattr(protocols, action.operation),
            parameters=action,
            mode=mode,
            update=update,
        )
    return instance.history
