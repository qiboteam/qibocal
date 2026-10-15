"""Action execution tracker."""

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NewType, Union

import yaml
from qibo import Circuit
from qibolab import Platform

from qibocal.auto.serialize import _nested_list_to_tuples
from qibocal.calibration.calibration import QubitId, QubitPairId

from .. import protocols
from ..config import log
from .mode import ExecutionMode
from .operation import (
    BoundProtocol,
    Data,
    DummyPars,
    OperationId,
    Protocol,
    Results,
    dummy_operation,
)

Id = NewType("Id", str)
"""Action identifiers type."""

Targets = list[QubitId] | list[QubitPairId] | list[tuple[QubitId, ...]]
"""Elements to be calibrated by a single protocol."""

SINGLE_ACTION = "action.yml"
CIRCUIT = "circuit.json"


@dataclass
class Action:
    """Action specification in the runcard."""

    id: Id
    """Action unique identifier."""
    operation: OperationId
    """Operation to be performed by the executor."""
    targets: Targets | None = None
    """Local qubits (optional)."""
    update: bool = True
    """Runcard update mechanism."""
    parameters: dict[str, Any] | None = None
    """Input parameters, either values or provider reference."""

    def dump(self, path: Path):
        """Dump single action to yaml."""
        if self.parameters is not None:
            path.mkdir(parents=True, exist_ok=True)

            for param, value in self.parameters.items():
                if type(value) is Circuit:
                    circuit_path = path / CIRCUIT
                    circuit_path.write_text(json.dumps(value.raw), encoding="utf-8")
                    self.parameters[param] = str(circuit_path)

            (path / SINGLE_ACTION).write_text(
                yaml.safe_dump(asdict(self)), encoding="utf-8"
            )

    @classmethod
    def load(cls, path):
        """Load action from yaml."""
        return cls(**yaml.safe_load((path / SINGLE_ACTION).read_text(encoding="utf-8")))

    @classmethod
    def cast(cls, source: Union[dict, "Action"], operation: str | None = None):
        """Cast an action source to an action."""
        if isinstance(source, Action):
            return source

        if operation is not None:
            source["operation"] = operation

        return cls(**source)


@dataclass(frozen=True)
class TaskId:
    """Unique identifier for executed tasks."""

    id: Id
    iteration: int

    def __str__(self):
        """Coincise representation."""
        return f"{self.id}-{self.iteration}"


DEFAULT_NSHOTS = 100
"""Default number on shots when the platform is not provided."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    operation: Protocol

    def __post_init__(self):
        # validate parameters
        self.operation.parameters_type.load(self.action.parameters)

    @classmethod
    def load(cls, path: Path):
        action = Action.load(path)
        return cls(action=action, operation=getattr(protocols, action.operation))

    def dump(self, path):
        self.action.dump(path)

    @property
    def targets(self) -> Targets:
        """Protocol targets."""
        return self.action.targets

    @property
    def id(self) -> Id:
        """Task Id."""
        return self.action.id

    @property
    def parameters(self):
        """Inputs parameters for self.operation."""
        return self.operation.parameters_type.load(self.action.parameters)

    @property
    def update(self):
        """Local update parameter."""
        return self.action.update

    def run(
        self,
        mode: ExecutionMode,
        folder: Path,
        platform: Platform | None = None,
        targets: Targets | None = None,
    ) -> "Completed":
        from .execute import Executor

        if self.targets is None:
            self.action.targets = targets

        try:
            if platform is not None:
                if self.parameters.nshots is None:
                    self.action.parameters["nshots"] = platform.settings.nshots
                if self.parameters.relaxation_time is None:
                    self.action.parameters["relaxation_time"] = (
                        platform.settings.relaxation_time
                    )
            else:
                if self.parameters.nshots is None:
                    self.action.parameters["nshots"] = DEFAULT_NSHOTS

            operation: Protocol = self.operation
            parameters = self.parameters

        except (RuntimeError, AttributeError):
            operation = dummy_operation
            parameters = DummyPars()
        bound = operation(pars=parameters)
        completed = Completed(self, folder, bound=bound)
        completed.dump_parameters()
        executor = Executor(platform, targets=self.targets)

        if ExecutionMode.ACQUIRE in mode:
            acquired = executor.acquire(bound)
            completed = replace(acquired, task=self, path=folder)
            completed.dump_data()
        if ExecutionMode.FIT in mode and operation.fit is not None:
            completed = executor.fit(completed)
            completed.dump_results()
        return completed


@dataclass
class Completed:
    """A complete or partial protocol execution, optionally backed by a task."""

    task: Task | None = None
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added
    """
    path: Path | None = None
    """Optional folder containing data and results files for task."""
    _data: Data | None = None
    """Protocol data."""
    _results: Results | None = None
    """Fitting output."""
    data_time: float = 0
    """Protocol data."""
    results_time: float = 0
    """Fitting output."""
    bound: BoundProtocol | None = None
    """Bound protocol used for this execution."""
    reports: Any = None
    """Report output, keyed by target for per-target callbacks."""
    protocol_id: str | None = None
    """Identifier for the protocol that was executed."""
    error: Exception | None = None
    """Error that occurred during execution, if any."""
    success: bool = True
    """Whether execution completed successfully."""
    _targets: Targets | None = None
    """Acquisition targets for data without target metadata."""

    def __post_init__(self):
        if self.task is not None:
            self.task = copy.deepcopy(self.task)
            if self.bound is None:
                self.bound = self.task.operation(pars=self.task.parameters)

    @property
    def targets(self) -> Targets:
        """Targets with acquired data, or the recorded acquisition selection."""
        data = self.data
        if isinstance(data, Data) and hasattr(data, "data"):
            if self.bound is not None and self.bound.protocol.two_qubit_gates:
                return data.pairs
            return data.qubits
        if self._targets is not None:
            return list(self._targets)
        if self.task is not None and self.task.targets is not None:
            return self.task.targets
        raise ValueError("Completed execution has no target metadata")

    @property
    def data(self) -> Data:
        """Access task's data."""
        if self._data is None:
            if self.bound is None or self.path is None:
                raise ValueError("Completed execution has no acquisition data")
            Data = self.bound.protocol.data_type
            self._data = Data.load(self.path)
            assert self._data is not None
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def results(self):
        """Access task's results."""
        if self._results is None and self.path is not None and self.bound is not None:
            Results = self.bound.protocol.results_type
            if Results is not None:
                self._results = Results.load(self.path)
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def dump_parameters(self):
        """Dump parameters."""
        if self.task is None or self.path is None:
            raise ValueError("Saving parameters requires a task and output path")
        self.task.dump(self.path)

    def dump_data(self):
        """Dumping data."""
        if self._data is not None:
            if self.path is None:
                raise ValueError("Saving data requires an output path")
            self._data.save(self.path)

    def dump_results(self):
        """Dumping results."""
        if self._results is not None:
            if self.path is None:
                raise ValueError("Saving results requires an output path")
            self._results.save(self.path)

    @classmethod
    def load(cls, path: Path):
        """Loading completed from path."""

        task = Task.load(path)
        return cls(path=path, task=task)

    def update_platform(self, platform: Platform):
        """Perform update on platform' parameters by looping over qubits or
        pairs."""
        if self.bound is None or self.bound.protocol.update is None:
            raise ValueError("Protocol does not support updating")
        for qubit in _nested_list_to_tuples(self.targets):
            try:
                self.bound.protocol.update(self.results, platform, qubit)
            except KeyError:
                log.warning(f"Skipping update of qubit {qubit} due to error in fit.")
