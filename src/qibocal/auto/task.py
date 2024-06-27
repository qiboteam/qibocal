"""Action execution tracker."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qibolab.platform import Platform

from qibocal import protocols

from ..config import log
from .mode import ExecutionMode
from .operation import Data, DummyPars, Results, Routine, dummy_operation
from .runcard import Action, Id, Targets


@dataclass
class TaskId:
    """Unique identifier for executed tasks."""

    id: Id
    iteration: int


DEFAULT_NSHOTS = 100
"""Default number on shots when the platform is not provided."""
PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    operation: Routine

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
        platform: Optional[Platform] = None,
        targets: Optional[Targets] = None,
        mode: Optional[ExecutionMode] = None,
    ):
        if self.targets is None:
            self.action.targets = targets

        completed = Completed(self)

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

            operation: Routine = self.operation
            parameters = self.parameters

        except (RuntimeError, AttributeError):
            operation = dummy_operation
            parameters = DummyPars()

        if ExecutionMode.ACQUIRE in mode:
            if operation.platform_dependent and operation.targets_dependent:
                completed.data, completed.data_time = operation.acquisition(
                    parameters,
                    platform=platform,
                    targets=self.targets,
                )

            else:
                completed.data, completed.data_time = operation.acquisition(
                    parameters, platform=platform
                )
        if ExecutionMode.FIT in mode:
            completed.results, completed.results_time = operation.fit(completed.data)
        return completed


@dataclass
class Completed:
    """A completed task."""

    task: Task
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added
    """
    path: Optional[Path] = None
    """Folder contaning data and results file for task."""
    _data: Optional[Data] = None
    """Protocol data."""
    _results: Optional[Results] = None
    """Fitting output."""
    data_time: float = 0
    """Protocol data."""
    results_time: float = 0
    """Fitting output."""

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)

    @property
    def data(self):
        """Access task's data."""
        if self._data is None:
            Data = self.task.operation.data_type
            self._data = Data.load(self.path)
        return self._data

    @property
    def results(self):
        """Access task's results."""
        if self._results is None:
            Results = self.task.operation.results_type
            self._results = Results.load(self.path)
        return self._results

    def dump(self):
        """Dump object to disk."""
        if self.path is None:
            raise ValueError("No known path where to dump execution results.")

        self.task.dump(self.path)
        if self._data is not None:
            self._data.save(self.path)
        if self._results is not None:
            self._results.save(self.path)

    def flush(self):
        """Dump disk, and release from memory."""
        self.dump()
        self._data = None
        self._results = None

    @classmethod
    def load(cls, path: Path):
        """Loading completed from path."""

        task = Task.load(path)
        return cls(path=path, task=task)

    def update_platform(self, platform: Platform):
        """Perform update on platform' parameters by looping over qubits or
        pairs."""
        for qubit in self.task.targets:
            try:
                self.task.operation.update(self.results, platform, qubit)
            except KeyError:
                log.warning(f"Skipping update of qubit {qubit} due to error in fit.")
        (self.path / PLATFORM_DIR).mkdir(parents=True, exist_ok=True)
        dump_platform(platform, self.path / PLATFORM_DIR)
