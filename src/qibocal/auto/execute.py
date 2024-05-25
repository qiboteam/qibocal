"""Tasks execution."""

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from qibolab import create_platform
from qibolab.platform import Platform

from qibocal import protocols
from qibocal.config import log, raise_error

from .history import History
from .mode import ExecutionMode
from .operation import Routine
from .runcard import Action, Runcard, Targets
from .task import Completed, Task


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
    output: Path
    """Output path."""
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
    def create(
        cls,
        name: str,
        platform: Union[Platform, str, None] = None,
        output: Optional[os.PathLike] = None,
    ):
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
        output = Path(output) if output is not None else Path.cwd()
        return cls(
            name=name,
            actions=[],
            history=History(),
            output=output,
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
        if isinstance(parameters, dict):
            parameters["operation"] = str(protocol)
            parameters.setdefault("id", len(self.history))
            parameters.setdefault("parameters", {})
            action = Action(**parameters)
        else:
            action = parameters
        task = Task(action, protocol)
        if isinstance(mode, ExecutionMode):
            log.info(
                f"Executing mode {mode.name if mode.name is not None else 'AUTOCALIBRATION'} on {task.id}."
            )

        if ExecutionMode.ACQUIRE in mode and task.id in self.history:
            raise_error(KeyError, f"{task.id} already contains acquisition data.")
        if ExecutionMode.FIT is mode and self.history[task.id]._results is not None:
            raise_error(KeyError, f"{task.id} already contains fitting results.")

        completed = task.run(
            platform=self.platform,
            targets=self.targets,
            folder=self.output,
            mode=mode,
        )

        if ExecutionMode.FIT in mode and self.platform is not None:
            completed.update_platform(platform=self.platform, update=self.update)

        self.history.push(completed)
        completed.dump(self.output)

        return completed


def run(runcard: Runcard, output: Path, mode: ExecutionMode):
    """Run runcard and dump to output."""
    platform = runcard.platform_obj
    targets = runcard.targets if runcard.targets is not None else list(platform.qubits)
    instance = Executor(
        history=History.load(output),
        platform=platform,
        targets=targets,
        output=output,
        update=runcard.update,
    )

    for action in runcard.actions:
        instance.run_protocol(
            protocol=getattr(protocols, action.operation),
            parameters=action,
            mode=mode,
        )
    return instance.history
