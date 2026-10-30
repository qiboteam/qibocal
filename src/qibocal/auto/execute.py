"""Tasks execution."""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter
from qibo.backends import construct_backend
from qibolab import Platform

from .. import protocols
from ..calibration import CalibrationPlatform, create_calibration_platform
from .history import History
from .notes import Note
from .operation import (
    DEFAULT_PARENT_PARAMETERS,
    BoundProtocol,
    Parameters,
)
from .output import Metadata, Output
from .task import Action, Completed, Targets, Task


def check_overlap_in_input_qubits(targets: np.typing.ArrayLike):
    """Check that target qubits do not contain duplicates."""

    targ = np.asarray(targets)
    if np.unique(targ).size != targ.size:
        raise ValueError("One or more target qubits were repeated.")


def _resolve(protocol: BoundProtocol, kwargs: dict) -> BoundProtocol:
    """Apply execution parameter overrides without mutating a bound protocol."""
    if not isinstance(protocol, BoundProtocol):
        raise TypeError("Execution requires a BoundProtocol")
    overrides = {
        name: value
        for name, value in kwargs.items()
        if name in DEFAULT_PARENT_PARAMETERS
    }
    binding = {name: value for name, value in kwargs.items() if name not in overrides}
    if binding:
        raise TypeError("Cannot pass binding arguments with a BoundProtocol")
    bound = protocol
    if not overrides:
        return bound
    parameters = copy(bound.parameters)
    for name, value in overrides.items():
        if not isinstance(parameters, Parameters) and not hasattr(parameters, name):
            raise TypeError(f"Protocol parameters do not support overriding {name}")
        setattr(parameters, name, value)
    return replace(bound, parameters=parameters)


def _invoke(callback: Callable, argument: object, **context: Any) -> Any:
    """Pass only explicitly declared context, including positional-only parameters."""
    parameters = list(signature(callback).parameters.values())[1:]
    args = [argument]
    kwargs = {}
    for parameter in parameters:
        if parameter.kind == Parameter.POSITIONAL_ONLY:
            if parameter.name in context:
                args.append(context[parameter.name])
            elif parameter.default is not Parameter.empty:
                args.append(parameter.default)
            else:
                raise TypeError(
                    f"Missing required positional-only argument: {parameter.name}"
                )
        elif (
            parameter.kind
            in (
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
            )
            and parameter.name in context
        ):
            kwargs[parameter.name] = context[parameter.name]
    return callback(*args, **kwargs)


def _runcard_value(value: Any) -> Any:
    """Convert a parameter value to the plain types found in runcards.

    Parameters may hold values normalized after loading, e.g. enumerations, ranges
    with NumPy scalars, or arrays computed during acquisition, which YAML cannot
    represent.
    """
    if isinstance(value, Enum):
        return _runcard_value(value.value)
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_runcard_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _runcard_value(item) for key, item in value.items()}
    return value


def _action_parameters(parameters: Any) -> dict[str, Any]:
    """Serializable parameters of a bound protocol, as found in runcards."""
    if not is_dataclass(parameters):
        raise TypeError(f"Cannot record parameters of type {type(parameters)}")
    values = {
        field.name: getattr(parameters, field.name) for field in fields(parameters)
    }
    for name in DEFAULT_PARENT_PARAMETERS:
        if hasattr(parameters, name):
            values[name] = getattr(parameters, name)
    return {name: _runcard_value(value) for name, value in values.items()}


class Executor:
    """Protocol executor for direct execution of protocols.

    Acquisition requires a BoundProtocol without requiring protocol registration.
    Subsequent phases consume and produce Completed instances.
    Context is passed to explicitly named callback parameters.
    Calibration output, history and lifecycle management are optional.
    Direct acquisition requires caller-managed platform connections.

    Once its output folder is initialized (cf. :meth:`open` and :meth:`__enter__`),
    the executor records the complete executions of built-in protocols
    (:meth:`__call__`) in its history and output folder, as soon as executed.
    """

    def __init__(
        self,
        platform: Platform | None,
        *,
        history: History | None = None,
        targets: Targets | None = None,
        update: bool = True,
        path: os.PathLike | None = None,
        meta: Metadata | None = None,
    ):
        """Initialize the executor with a platform.

        Args:
            platform: The hardware platform to use for acquisition and updates.
            history: Calibration execution history.
            targets: Optional default calibration targets. If omitted, targets
                must be supplied for each protocol invocation.
            update: Whether to apply fitted results to the platform.
            path: Optional calibration output directory.
            meta: Calibration execution metadata.
        """
        self.platform = platform
        self.history = history if history is not None else History()
        self.targets = (
            TypeAdapter(Targets).validate_python(targets)
            if targets is not None
            else None
        )
        self._update_enabled = update
        self.path = Path(path) if path is not None else None
        self.meta = meta
        self.notes: list[Note] = []
        self._initialized = False
        if self.targets is not None:
            check_overlap_in_input_qubits(self.targets)

    @classmethod
    def create(
        cls,
        path: os.PathLike,
        targets: Targets | None = None,
        platform: CalibrationPlatform | Platform | str | None = None,
        **kwargs: Any,
    ) -> Executor:
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

    def _init(self, force: bool = False):
        """Initialize execution once and connect the platform."""
        if self.path is None or self.meta is None or self.platform is None:
            raise ValueError(
                "Calibration initialization requires an output path, metadata and platform"
            )
        if not self._initialized:
            # generate output folder
            path = Output.mkdir(self.path, force)

            # generate meta
            output = Output(History(), self.meta, self.platform, notes=self.notes)
            output.dump(path)

            # start timer
            self.meta.start()
            self._initialized = True

        # connect and initialize platform
        self.platform.connect()

    def close(self):
        """Close execution."""
        if self.path is None or self.meta is None or self.platform is None:
            raise ValueError(
                "Calibration finalization requires an output path, metadata and platform"
            )

        # stop and disconnect platform
        self.platform.disconnect()

        self.meta.end()

        # dump history, metadata, and updated platform
        output = Output(self.history, self.meta, self.platform, notes=self.notes)
        output.dump(self.path)

    @classmethod
    @contextmanager
    def open(
        cls,
        path: os.PathLike,
        targets: Targets | None = None,
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
        ex._init(force)

        try:
            yield ex
        finally:
            ex.close()

    def __enter__(self):
        """Enter or reenter the execution context.

        Initialize calibration output on first entry and reconnect the platform
        on subsequent entries.
        """
        self._init()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit execution context.

        This pairs with :meth:`__enter__`.
        """
        self.close()
        return False

    def __call__(
        self,
        bound: BoundProtocol,
        skip_fit: bool = False,
        **kwargs: Any,
    ) -> Completed:
        """Execute a complete protocol workflow.

        Executes acquisition, optionally fit, optionally report, and optionally update.

        Args:
            bound: The bound protocol to execute.
            skip_fit: If True, skip the fit phase.
            **kwargs: Execution overrides, including
                targets and acquisition settings. The platform cannot be overridden.

        Returns:
            A Completed object with results and metadata.
        """
        completed = self.acquire(bound, **kwargs)
        recorded = self._dump_task(completed) if self._initialized else None
        recording = recorded is not None
        if recording:
            completed = recorded
        if not skip_fit and bound.protocol.fit is not None:
            completed = self.fit(completed)
            if recording:
                completed.dump_results()
        if completed.results is not None and bound.protocol.report is not None:
            completed = self.report(completed)
        if (
            completed.results is not None
            and bound.protocol.update is not None
            and self._update_enabled
        ):
            completed = self.update(completed)
        if recording:
            self.history.push(completed)
            self.history.dump(self.path)
        return completed

    def _dump_task(self, completed: Completed) -> Completed | None:
        """Dump an acquired execution as a new task of the output folder.

        Only registered protocols are recorded, since only those can be loaded back
        from the output folder. Returns the execution bound to its task, or `None`
        if not recorded.
        """
        protocol = completed.bound.protocol
        name = next(
            (name for name, item in protocols.PROTOCOLS.items() if item is protocol),
            None,
        )
        if name is None:
            warnings.warn(
                "Execution not recorded: the protocol is not registered in "
                "qibocal.protocols",
                stacklevel=3,
            )
            return None
        try:
            action = Action(
                id=name,
                operation=name,
                targets=completed.targets,
                update=self._update_enabled,
                parameters=_action_parameters(completed.bound.parameters),
            )
        except (TypeError, ValueError) as error:
            warnings.warn(f"Execution not recorded: {error}", stacklevel=3)
            return None
        task = Task(action=action, operation=protocol)
        folder = self.history.task_path(
            self.history._pending_task_id(task.id), self.path
        )
        recorded = replace(completed, task=task, path=folder)
        recorded.dump_parameters()
        recorded.dump_data()
        return recorded

    def _context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        targets = kwargs.pop("targets", None)
        targets = self.targets if targets is None else targets
        if targets is None:
            raise ValueError(
                "Targets must be supplied when the executor has no default targets"
            )
        selected = TypeAdapter(Targets).validate_python(targets)
        check_overlap_in_input_qubits(selected)
        return {
            "platform": self.platform,
            "targets": selected,
        }

    def acquire(
        self,
        bound: BoundProtocol,
        **kwargs: Any,
    ) -> Completed:
        """Execute only the acquisition phase.

        Args:
            bound: The bound protocol to acquire data from.
            **kwargs: Execution overrides, including
                targets and acquisition settings. The platform cannot be overridden.

        Returns:
            A new Completed instance retaining data and the bound protocol.
        """
        targets = kwargs.pop("targets", None)
        bound = _resolve(bound, kwargs)
        context = self._context({"targets": targets})
        parameters = bound.parameters
        if isinstance(parameters, Parameters) and isinstance(
            context["platform"], Platform
        ):
            defaults = {
                name: getattr(context["platform"].settings, name)
                for name in DEFAULT_PARENT_PARAMETERS
                if getattr(parameters, name, None) is None
            }
            if defaults:
                parameters = copy(parameters)
                for name, value in defaults.items():
                    setattr(parameters, name, value)
                bound = replace(bound, parameters=parameters)
        start = time.perf_counter()
        data = _invoke(bound.protocol.acquisition, parameters, **context)
        return Completed(
            bound=bound,
            _data=data,
            _targets=context["targets"],
            data_time=time.perf_counter() - start,
        )

    def _completed_context(
        self, completed: Completed, targets: Targets | None
    ) -> tuple[BoundProtocol, dict[str, Any]]:
        if not isinstance(completed, Completed):
            raise TypeError("Downstream execution requires a Completed instance")
        if completed.bound is None:
            raise ValueError("Completed execution has no bound protocol")
        context = (
            {"platform": self.platform, "targets": completed.targets}
            if targets is None
            else self._context({"targets": targets})
        )
        return completed.bound, context

    def fit(
        self,
        completed: Completed,
        *,
        targets: Targets | None = None,
    ) -> Completed:
        """Execute the fit phase on existing data.

        This can be used to re-fit data that was previously acquired,
        or to apply a different fitting strategy.

        Args:
            completed: An acquired execution, optionally loaded from disk.
            targets: Optional override of the targets recorded at acquisition.

        Returns:
            A new Completed instance with fitting results attached.

        Raises:
            ValueError: If the protocol does not have a fit function.
        """
        bound, context = self._completed_context(completed, targets)
        if bound.protocol.fit is None:
            raise ValueError("Protocol does not support fitting")

        start = time.perf_counter()
        results = _invoke(
            bound.protocol.fit,
            completed.data,
            fitpars=bound.fitpars,
            **context,
        )
        return replace(
            completed,
            _results=results,
            results_time=time.perf_counter() - start,
            reports=None,
        )

    def report(
        self,
        completed: Completed,
        *,
        targets: Targets | None = None,
    ) -> Completed:
        """Execute the report phase.

        Args:
            completed: An acquired or fitted execution.
            targets: Optional override of the targets recorded at acquisition.

        Returns:
            A new Completed instance with report output attached.

        Raises:
            ValueError: If the protocol does not have a report function.
        """
        bound, context = self._completed_context(completed, targets)
        if bound.protocol.report is None:
            raise ValueError("Protocol does not support reporting")

        context |= {
            "fit": completed.results,
            "results": completed.results,
            "reportpars": bound.reportpars,
        }
        if "target" in signature(bound.protocol.report).parameters:
            reports = {
                target: _invoke(
                    bound.protocol.report, completed.data, target=target, **context
                )
                for target in context["targets"]
            }
        else:
            reports = _invoke(bound.protocol.report, completed.data, **context)
        return replace(completed, reports=reports)

    def update(
        self,
        completed: Completed,
        *,
        targets: Targets | None = None,
    ) -> Completed:
        """Execute the update phase to apply results to platform.

        Args:
            completed: A fitted execution.
            targets: Optional override of the targets recorded at acquisition.

        Returns:
            A new Completed instance after applying the platform update.

        Raises:
            ValueError: If the protocol has no update function or no platform
                is configured.
        """
        bound, context = self._completed_context(completed, targets)
        if bound.protocol.update is None:
            raise ValueError("Protocol does not support updating")
        if context["platform"] is None:
            raise ValueError("Executor does not have a platform configured")

        parameters = signature(bound.protocol.update).parameters
        if "target" in parameters or "qubit" in parameters:
            for target in context["targets"]:
                _invoke(
                    bound.protocol.update,
                    completed.results,
                    target=target,
                    qubit=target,
                    **context,
                )
        else:
            _invoke(bound.protocol.update, completed.results, **context)
        return replace(completed)
