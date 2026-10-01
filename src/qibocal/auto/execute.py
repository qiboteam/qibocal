"""Tasks execution."""

from __future__ import annotations

import operator
import os
from collections.abc import Callable
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import fields
from functools import cached_property, reduce
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter
from qibo.backends import construct_backend
from qibolab import Platform

from qibocal import protocols
from qibocal.config import log

from ..calibration import CalibrationPlatform, create_calibration_platform
from .history import History
from .mode import AUTOCALIBRATION, ExecutionMode
from .operation import (
    DEFAULT_PARENT_PARAMETERS,
    BoundProtocol,
    Parameters,
    Protocol,
    ProtocolsCollection,
)
from .operation import Completed as ProtocolCompleted
from .output import PLATFORM, Metadata, Output
from .task import Action, Completed, Targets, Task


def check_overlap_in_input_qubits(targets: np.typing.ArrayLike):
    """Check that target qubits do not contain duplicates."""

    targ = np.asarray(targets)
    if np.unique(targ).size != targ.size:
        raise ValueError("One or more target qubits were repeated.")


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


class Executor:
    """Protocol executor for direct execution of bound protocols.

    This executor executes BoundProtocol instances without requiring
    protocol registration. Context is passed to explicitly named callback parameters.
    Calibration output, history and lifecycle management are optional.
    Direct acquisition requires caller-managed platform connections.
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
        sources: list[ProtocolsCollection] | None = None,
    ):
        """Initialize the executor with a platform.

        Args:
            platform: The hardware platform to use for acquisition and updates.
            history: Calibration execution history.
            targets: Default calibration targets.
            update: Whether to apply fitted results to the platform.
            path: Optional calibration output directory.
            meta: Calibration execution metadata.
            sources: Additional calibration protocols.
        """
        self.platform = platform
        self.history = history if history is not None else History()
        self.targets = TypeAdapter(Targets).validate_python(
            targets if targets is not None else []
        )
        self._update_enabled = update
        self.path = Path(path) if path is not None else None
        self.meta = meta
        self.sources = sources if sources is not None else []
        check_overlap_in_input_qubits(self.targets)

        if self.path is not None:
            for name, protocol in self.protocols.items():
                setattr(self, name, self._wrapped_protocol(protocol, name))

    def __call__(
        self,
        bound: BoundProtocol,
        skip_fit: bool = False,
        *,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> ProtocolCompleted:
        """Execute a complete protocol workflow.

        Executes acquisition, optionally fit, optionally report, and optionally update.

        Args:
            bound: The bound protocol to execute.
            skip_fit: If True, skip the fit phase.
            platform: Override the executor platform for all phases.
            targets: Override the executor targets for all phases.

        Returns:
            A Completed object with results and metadata.
        """
        # Execute acquisition
        context = self._context(platform, targets)
        data = self.acquire(bound, **context)

        # Execute fit if not skipped and available
        if skip_fit or bound.protocol.fit is None:
            results = None
        else:
            results = self.fit(data, bound, **context)

        # Execute report if available
        reports = None
        if results is not None and bound.protocol.report is not None:
            reports = self.report(data, results, bound, **context)

        # Execute update if available
        if (
            results is not None
            and bound.protocol.update is not None
            and self._update_enabled
        ):
            self.update(results, bound, **context)

        return ProtocolCompleted(
            data=data,
            results=results,
            success=True,
            reports=reports,
        )

    def _context(
        self, platform: Platform | None, targets: Targets | None
    ) -> dict[str, Any]:
        selected = TypeAdapter(Targets).validate_python(
            self.targets if targets is None else targets
        )
        check_overlap_in_input_qubits(selected)
        return {
            "platform": self.platform if platform is None else platform,
            "targets": selected,
        }

    def acquire(
        self,
        bound: BoundProtocol,
        *,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> object:
        """Execute only the acquisition phase.

        Args:
            bound: The bound protocol to acquire data from.
            platform: Override the executor platform.
            targets: Override the executor targets.

        Returns:
            The data object returned by the acquisition function.
        """
        context = self._context(platform, targets)
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
        return _invoke(bound.protocol.acquisition, parameters, **context)

    def fit(
        self,
        data: object,
        bound: BoundProtocol,
        *,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> object:
        """Execute the fit phase on existing data.

        This can be used to re-fit data that was previously acquired,
        or to apply a different fitting strategy.

        Args:
            data: Data from acquisition (or loaded from disk).
            bound: The bound protocol with fit parameters.
            platform: Override the executor platform.
            targets: Override the executor targets.

        Returns:
            The results object returned by the fit function.

        Raises:
            ValueError: If the protocol does not have a fit function.
        """
        if bound.protocol.fit is None:
            raise ValueError("Protocol does not support fitting")

        return _invoke(
            bound.protocol.fit,
            data,
            fitpars=bound.fitpars,
            fit_params=bound.fitpars,
            **self._context(platform, targets),
        )

    def report(
        self,
        data: object,
        results: object,
        bound: BoundProtocol,
        *,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> Any:
        """Execute the report phase.

        Args:
            data: Data from acquisition.
            results: Results from fitting.
            bound: The bound protocol with report parameters.
            platform: Override the executor platform.
            targets: Override the executor targets.

        Returns:
            Callback output, or a target-to-output mapping for per-target reports.

        Raises:
            ValueError: If the protocol does not have a report function.
        """
        if bound.protocol.report is None:
            raise ValueError("Protocol does not support reporting")

        context = self._context(platform, targets) | {
            "fit": results,
            "results": results,
            "reportpars": bound.reportpars,
            "report_params": bound.reportpars,
        }
        if "target" in signature(bound.protocol.report).parameters:
            return {
                target: _invoke(bound.protocol.report, data, target=target, **context)
                for target in context["targets"]
            }
        return _invoke(bound.protocol.report, data, **context)

    def update(
        self,
        results: object,
        bound: BoundProtocol,
        *,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> None:
        """Execute the update phase to apply results to platform.

        Args:
            results: Results from fitting.
            bound: The bound protocol defining the update function.
            platform: Override the executor platform.
            targets: Override the executor targets.

        Raises:
            ValueError: If the protocol has no update function or no platform
                is configured.
        """
        if bound.protocol.update is None:
            raise ValueError("Protocol does not support updating")
        context = self._context(platform, targets)
        if context["platform"] is None:
            raise ValueError("Executor does not have a platform configured")

        parameters = signature(bound.protocol.update).parameters
        if "target" in parameters or "qubit" in parameters:
            for target in context["targets"]:
                _invoke(
                    bound.protocol.update,
                    results,
                    target=target,
                    qubit=target,
                    **context,
                )
        else:
            _invoke(bound.protocol.update, results, **context)

    @cached_property
    def protocols(self) -> ProtocolsCollection:
        return reduce(operator.or_, [protocols.PROTOCOLS] + self.sources)

    def run_protocol(
        self,
        protocol: Protocol,
        parameters: Action,
        mode: ExecutionMode = AUTOCALIBRATION,
    ) -> Completed:
        """Run a calibration protocol and record the completed task.

        The executor preserves the execution history and chooses the platform
        instance used for the task based on the requested mode. If acquisition is
        requested, the current live platform is used. If only fitting or analysis
        is required, the platform is reconstructed from the output folder so that
        the exact experiment configuration is reused.
        If the mode contains :class:`ExecutionMode.FIT`, and the action is
        configured to update the platform, it is updated using the fitted parameters.
        """

        output = self.path
        if output is None or self.platform is None:
            raise ValueError(
                "Calibration execution requires an output path and platform"
            )

        task = Task(action=parameters, operation=protocol)
        log.info(f"Executing mode {mode} on {task.action.id}.")
        completed = task.run(
            platform=(
                # if data acquisition is required, the platform must be created
                # from scratch with all its hardware configurations;
                # when the executor is only fitting, the exact same platform of
                # the experiment (saved in the experiment folder) is recreated,
                # and the hardware configuration is unnecessary.
                self.platform
                if ExecutionMode.ACQUIRE in mode
                else CalibrationPlatform.from_datafolder(
                    folder_path=output / PLATFORM,
                    platform_name=self.platform.name,
                )
            ),
            targets=self.targets,
            mode=mode,
            folder=self.history.task_path(
                self.history._pending_task_id(task.id), output
            ),
        )
        self.history.push(completed)

        # TODO: drop, as the conditions won't be necessary any longer, and then it could
        # be performed as part of `task.run` https://github.com/qiboteam/qibocal/issues/910
        if (
            ExecutionMode.FIT in mode
            and self._update_enabled
            and task.update
            and protocol.update is not None
            and completed.results is not None
        ):
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
            return self.run_protocol(protocol, parameters=action, mode=mode)

        return wrapper

    @classmethod
    def create(
        cls,
        path: os.PathLike,
        targets: Targets,
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

    def init(self, force: bool = False):
        """Initialize execution."""
        if self.path is None or self.meta is None or self.platform is None:
            raise ValueError(
                "Calibration initialization requires an output path, metadata and platform"
            )
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
        if self.path is None or self.meta is None or self.platform is None:
            raise ValueError(
                "Calibration finalization requires an output path, metadata and platform"
            )

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
        if self.platform is None:
            raise ValueError("Executor does not have a platform configured")
        self.platform.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit execution context.

        This pairs with :meth:`__enter__`.
        """
        self.close()
        return False
