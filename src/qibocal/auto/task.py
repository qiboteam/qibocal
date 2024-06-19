"""Action execution tracker."""

import copy
from dataclasses import dataclass
from pathlib import Path
from statistics import mode
from typing import Optional

from qibolab.platform import Platform
from qibolab.serialize import dump_platform

from qibocal import protocols

from ..config import log
from .mode import ExecutionMode
from .operation import Data, DummyPars, Results, Routine, dummy_operation
from .runcard import Action, Id, Targets

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed. But not so insanely big not to fit in a native integer."""
DEFAULT_NSHOTS = 100
"""Default number on shots when the platform is not provided."""
TaskId = tuple[Id, int]
"""Unique identifier for executed tasks."""
PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    operation: Routine

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
        platform: Platform = None,
        targets: Targets = list,
        mode: ExecutionMode = None,
        folder: Path = None,
    ):

        if self.targets is None:
            self.action.targets = targets

        completed = Completed(self, folder)

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
    folder: Path
    """Folder with data and results."""
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
    def datapath(self):
        """Path contaning data and results file for task."""
        path = self.folder / "data" / f"{self.task.id}"
        if not path.is_dir():
            path.mkdir(parents=True)
        return path

    @property
    def results(self):
        """Access task's results."""
        if self._results is None:
            Results = self.task.operation.results_type
            self._results = Results.load(self.datapath)
        return self._results

    @results.setter
    def results(self, results: Results):
        """Set and store results."""
        self._results = results
        self._results.save(self.datapath)

    @property
    def data(self):
        """Access task's data."""
        if self._data is None:
            Data = self.task.operation.data_type
            self._data = Data.load(self.datapath)
        return self._data

    @data.setter
    def data(self, data: Data):
        """Set and store data."""
        self._data = data
        self._data.save(self.datapath)

    def dump(self, path):
        """test"""
        self.task.dump(self.datapath)

    @classmethod
    def load(cls, folder: Path):
        """Loading completed from path."""

        task = Task.load(folder)
        return cls(task=task, folder=folder.parents[1])

    def update_platform(self, platform: Platform, update: bool):
        """Perform update on platform' parameters by looping over qubits or pairs."""
        if self.task.update and update:
            for qubit in self.task.targets:
                try:
                    self.task.operation.update(self.results, platform, qubit)
                except KeyError:
                    log.warning(
                        f"Skipping update of qubit {qubit} due to error in fit."
                    )
            (self.datapath / PLATFORM_DIR).mkdir(parents=True, exist_ok=True)
            dump_platform(platform, self.datapath / PLATFORM_DIR)
