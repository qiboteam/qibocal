"""Tasks execution."""

import importlib
import importlib.util
import operator
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
from functools import cached_property, reduce
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from qibo.backends import construct_backend
from qibolab import Platform

from qibocal import protocols
from qibocal.config import log

from ..calibration import CalibrationPlatform, create_calibration_platform
from .history import History
from .mode import AUTOCALIBRATION, ExecutionMode
from .operation import Protocol, ProtocolsCollection
from .output import Metadata, Output
from .task import Action, Completed, Targets, Task

PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


def check_overlap_in_input_qubits(targets: np.typing.ArrayLike):
    """Check that target qubits do not contain duplicates."""

    targ = np.asarray(targets)
    if np.unique(targ).size != targ.size:
        raise ValueError("One or more target qubits were repeated.")


class Executor(BaseModel):
    """Execute a tasks' graph and tracks its history."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    history: History
    """The execution history, with results and exit states."""
    targets: Targets
    """Qubits/Qubit Pairs to be calibrated."""
    platform: CalibrationPlatform
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""
    path: Path
    meta: Metadata

    sources: list[ProtocolsCollection] = Field(default_factory=list)
    """Sources to extend core protocol set."""

    def model_post_init(self, context: Any) -> None:
        """Register protocols for execution."""
        # explicitly unused
        _ = context

        for name, protocol in self.protocols.items():
            object.__setattr__(self, name, self._wrapped_protocol(protocol, name))

        check_overlap_in_input_qubits(self.targets)

    @cached_property
    def protocols(self) -> ProtocolsCollection:
        return reduce(operator.or_, [protocols.PROTOCOLS] + self.sources)

    def run_protocol(
        self,
        protocol: Protocol,
        parameters: Action,
        mode: ExecutionMode = AUTOCALIBRATION,
        output: Path | None = None,
    ) -> Completed:
        """Run single protocol in ExecutionMode mode."""

        task = Task(action=parameters, operation=protocol)
        log.info(f"Executing mode {mode} on {task.action.id}.")
        completed = task.run(
            platform=self.platform,
            targets=self.targets,
            mode=mode,
            folder=self.history.task_path(
                self.history._pending_task_id(task.id), output
            ),
        )
        self.history.push(completed)

        # TODO: drop, as the conditions won't be necessary any longer, and then it could
        # be performed as part of `task.run` https://github.com/qiboteam/qibocal/issues/910
        if ExecutionMode.FIT in mode:
            if self.update and task.update:
                completed.update_platform(platform=self.platform)

        return completed

    def _wrapped_protocol(self, protocol: Protocol, operation: str):
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
            *args: Any,
            parameters: dict | None = None,
            id: str = operation,
            mode: ExecutionMode = AUTOCALIBRATION,
            update: bool = True,
            targets: Targets | None = None,
            **kwargs: Any,
        ):
            # casting targest to be of type Targets if not None
            if targets is not None:
                targets = TypeAdapter(Targets).validate_python(targets)
                # check if input is correct
                check_overlap_in_input_qubits(targets)

            positional = dict(
                zip((f.name for f in fields(protocol.parameters_type)), args)
            )
            params = deepcopy(parameters) if parameters is not None else {}
            action = Action.cast(
                source={
                    "id": id,
                    "operation": operation,
                    "targets": targets,
                    "update": update,
                    "parameters": params | positional | kwargs,
                }
            )
            return self.run_protocol(
                protocol, parameters=action, mode=mode, output=self.path
            )

        return wrapper

    @classmethod
    def create(
        cls,
        path: os.PathLike,
        targets: Targets,
        platform: CalibrationPlatform | Platform | str | None = None,
        **kwargs: Any,
    ) -> "Executor":
        """Create protocols' executor.

        This is a wrapper of the default constructor, which is only handling different
        platforms specification.

        For the full set of arguments, cf. :class:`Executor`.
        """
        platform = (
            platform
            if isinstance(platform, CalibrationPlatform)
            else CalibrationPlatform.from_platform(platform)
            if isinstance(platform, Platform)
            else create_calibration_platform(
                platform if isinstance(platform, str) else "mock"
            )
        )
        path_ = Path(path)
        backend = construct_backend(backend="qibolab", platform=platform)
        return cls(
            history=History(),
            platform=platform,
            path=path_,
            targets=targets,
            meta=Metadata.generate(backend),
            **kwargs,
        )

    def init(self, force: bool = False):
        """Initialize execution."""
        # generate output folder
        path = Output.mkdir(self.path, force)

        # generate meta
        output = Output(History(), self.meta, self.platform)
        output.dump(path)

        # start timer
        self.meta.start()

        # connect and initialize platform
        self.platform.connect()

    def close(self):
        """Close execution."""
        assert self.meta is not None and self.path is not None

        # stop and disconnect platform
        self.platform.disconnect()

        self.meta.end()

        # dump history, metadata, and updated platform
        output = Output(self.history, self.meta, self.platform)
        output.dump(self.path)

    @classmethod
    @contextmanager
    def open(
        cls,
        path: os.PathLike,
        targets: Targets,
        force: bool = False,
        platform: CalibrationPlatform | str | None = None,
        update: bool | None = None,
        **kwargs: Any,
    ):
        """Enter the execution context.

        For the full set of arguments, cf. :class:`Executor`.
        """
        if update is not None:
            kwargs["update"] = update

        ex = cls.create(path=path, platform=platform, targets=targets, **kwargs)
        ex.init(force)

        try:
            yield ex
        finally:
            ex.close()

    def __enter__(self):
        """Reenter the execution context.

        This method its here to reuse an already existing (and
        initialized) executor, in a new context.

        It should not be used with new executors. In which case, cf. :meth:`__open__`.
        """
        # connect and initialize platform
        self.platform.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit execution context.

        This pairs with :meth:`__enter__`.
        """
        self.close()
        return False


def _register(name: str, obj: Any) -> None:
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
    obj.__name__ = qualified
    obj.__spec__ = None
