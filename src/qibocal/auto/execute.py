"""Tasks execution."""

import sys
from dataclasses import dataclass
from typing import Optional, Union

from qibolab import create_platform
from qibolab.platform import Platform

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
    name: Optional[str] = None
    """Name, used just as a label but also to register the module."""

    def __post_init__(self):
        if self.name is not None:
            if self.name in sys.modules:
                raise ValueError(
                    f"Module '{self.name}' already present. "
                    "Choose a different one to avoid overwriting it."
                )
            sys.modules[name] = self

    def __getattribute__(self, name):
        """Provide access to routines through the executor.

        This is done mainly to support the import mechanics: the routines retrieved
        through the object will have it pre-registered.
        """
        modname = super().__getattribute__(name)
        if modname is None:
            # no module registration, immediately fall back
            return super().__getattribute__(name)

        attrs = {
            "__spec__": None,
            "__name__": modname,
        }

        try:
            # stage 1: module definition
            return attrs[name]
        except KeyError:
            pass
        try:
            # stage 2: routines look up
            return getattr(protocols, name)
        except AttributeError:
            # stage 3: fall back on regular attributes
            return super().__getattribute__(name)

    @classmethod
    def create(cls, name: str, platform: Union[Platform, str]):
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
