"""Tasks execution."""

import importlib
import importlib.util
import os
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


def _register(name, obj):
    # prevent overwriting existing modules
    if name in sys.modules:
        raise ValueError(
            f"Module '{name}' already present. "
            "Choose a different one to avoid overwriting it."
        )

    # allow relative paths, where relative is intended respect to package root
    root = __name__.split(".")[0]
    qualified = importlib.util.resolve_name(name, root)

    # allow to nest module in arbitrary subpackage
    if "." in qualified:
        parent_name, _, child_name = qualified.rpartition(".")
        parent_module = importlib.import_module(parent_name)
        setattr(parent_module, child_name, obj)

    sys.modules[qualified] = obj
    obj.name = obj.__name__ = qualified
    obj.__spec__ = None


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
        """Register as a module, if a name is specified."""
        if self.name is not None:
            _register(self.name, self)

    def __getattribute__(self, name: str):
        """Provide access to routines through the executor.

        This is done mainly to support the import mechanics: the routines retrieved
        through the object will have it pre-registered.
        """
        modname = super().__getattribute__("name")
        if modname is None:
            # no module registration, immediately fall back
            return super().__getattribute__(name)

        try:
            # routines look up
            if name.startswith("_"):
                # internal attributes should never be routines
                raise AttributeError

            protocol = getattr(protocols, name)
            return lambda *args, **kwargs: self.run_protocol(protocol, *args, **kwargs)
        except AttributeError:
            # fall back on regular attributes
            return super().__getattribute__(name)

    @classmethod
    def create(cls, name: str, platform: Union[Platform, str, None] = None):
        """Load list of protocols."""
        platform = (
            platform
            if isinstance(platform, Platform)
            else create_platform(
                platform
                if platform is not None
                else os.environ.get("QIBO_PLATFORM", "dummy")
            )
        )
        return cls(
            name=name,
            history=History(),
            platform=platform,
            targets=list(platform.qubits),
            update=True,
        )

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
