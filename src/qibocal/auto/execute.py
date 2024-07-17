"""Tasks execution."""

import importlib
import importlib.util
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

from qibo.backends import construct_backend
from qibolab import create_platform
from qibolab.platform import Platform

from qibocal import protocols
from qibocal.config import log

from .history import History
from .mode import AUTOCALIBRATION, ExecutionMode
from .operation import Routine
from .output import Metadata, Output
from .task import Action, Completed, Targets, Task

PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


def _register(name: str, obj):
    """Register object as module.

    With a small abuse of the Python module system, the object is registered as a
    module, with the given `name`.
    `name` may contain dots, cf. :attr:`Executor.name` for clarifications about their
    meaning.

    .. note::

        This is mainly used to register executors, such that the protocols can be
        bound to it through the `import` keyword, in order to construct an intuitive
        syntax, apparently purely functional, maintaining the context in a single
        `Executor` "global" object.
    """
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
    """Symbol for the executor.

    This can be used generically to distinguish the executor, but its specific use is to
    register a module with this name in `sys.modules`.
    They can contain dots, `.`, that are interpreted as usual by the Python module
    system.

    .. note::

        As a special case, mainly used for internal purposes, names starting with `.`
        are also allowed, and they are interpreted relative to this package (in the top
        scope).
    """

    def __post_init__(self):
        """Register as a module, if a name is specified."""
        if self.name is not None:
            _register(self.name, self)

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
        parameters: Action,
        mode: ExecutionMode = AUTOCALIBRATION,
    ) -> Completed:
        """Run single protocol in ExecutionMode mode."""
        task = Task(action=parameters, operation=protocol)
        log.info(f"Executing mode {mode} on {task.action.id}.")

        completed = task.run(platform=self.platform, targets=self.targets, mode=mode)
        self.history.push(completed)

        # TODO: drop, as the conditions won't be necessary any longer, and then it could
        # be performed as part of `task.run` https://github.com/qiboteam/qibocal/issues/910
        if ExecutionMode.FIT in mode:
            if self.update and task.update:
                completed.update_platform(platform=self.platform)

        return completed

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
            return self._wrapped_protocol(protocol, name)
        except AttributeError:
            # fall back on regular attributes
            return super().__getattribute__(name)

    def _wrapped_protocol(self, protocol: Routine, operation: str):
        """Create a bound protocol.

        Returns a closure, already wrapping the current `Executor` instance, but
        specific to the `protocol` chosen.
        The parameters of this wrapper function maps to protocol's ones, in particular:

            - the keyword argument `mode` is used as the execution mode (defaults to
              `AUTOCALIBRATION`)
            - the keyword argument `id` is used as the `id` for the given operation
              (defaults to `protocol` identifier, the same used to import and invoke
              it)

        then the protocol specific are resolved, with the following priority:

            - explicit keyword arguments have the highest priorities
            - items in the dictionary passed with the keyword `parameters`
            - positional arguments, which are associated to protocols parameters in the
              same order in which they are defined (and documented) in their respective
              parameters classes

        .. attention::

            Despite the priority being clear, it is advised to use only one of the
            former schemes to pass parameters, to avoid confusion due to unexpected
            overwritten arguments.

            E.g. for::

                resonator_spectroscopy(1e7, 1e5, freq_width=1e8)

            the `freq_width` will be `1e8`, and `1e7` will be silently overwritten and
            ignored (as opposed to a regular Python function, where a `TypeError` would
            be raised).

            The priority defined above is strictly and silently respected, so just pay
            attention during invocations.
        """

        def wrapper(
            *args,
            parameters: Optional[dict] = None,
            id: str = operation,
            mode: ExecutionMode = AUTOCALIBRATION,
            **kwargs,
        ):
            positional = dict(
                zip((f.name for f in fields(protocol.parameters_type)), args)
            )
            params = deepcopy(parameters) if parameters is not None else {}
            action = Action.cast(
                source={
                    "id": id,
                    "operation": operation,
                    "parameters": params | positional | kwargs,
                }
            )
            return self.run_protocol(protocol, parameters=action, mode=mode)

        return wrapper

    def unload(self):
        """Unlist the executor from available modules."""
        if self.name is not None:
            del sys.modules[self.name]

    def __del__(self):
        """Revert constructions side-effects.

        .. note::

            This is to make sure that the executor is properly unregistered from
            `sys.modules`. However, it is not reliable to be called directly, since `del
            executor` is not guaranteed to immediately invoke this method, cf. the note
            for :meth:`object.__del__`.
        """
        try:
            self.unload()
        except KeyError:
            # it has been explicitly unloaded, no need to do it again
            pass

    def init(
        self,
        folder: Path,
        force: bool = False,
        platform: Optional[Platform] = None,
        targets: Optional[Targets] = None,
    ):
        if platform is not None:
            self.platform = platform
        else:
            platform = self.platform

        backend = construct_backend(backend="qibolab", platform=platform)

        if targets is not None:
            self.targets = targets

        # generate output folder
        path = Output.mkdir(folder, force)

        # generate meta
        meta = Metadata.generate(path.name, backend)
        output = Output(History(), meta, platform)
        output.dump(path)

        # connect and initialize platform
        platform.connect()

        # run
        meta.start()

        return path, output, meta, platform
