"""Action execution tracker."""
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from qibolab.platform import Platform
from qibolab.qubits import QubitId

from ..protocols.characterization import Operation
from ..utils import allocate_qubits_pairs, allocate_single_qubits
from .mode import ExecutionMode
from .operation import (
    DATAFILE,
    RESULTSFILE,
    Data,
    DummyPars,
    Qubits,
    QubitsPairs,
    Results,
    Routine,
    dummy_operation,
)
from .runcard import Action, Id
from .status import Normal, Status

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed. But not so insanely big not to fit in a native integer."""

TaskId = tuple[Id, int]
"""Unique identifier for executed tasks."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    iteration: int = 0
    """Task iteration (to be used for the ExceptionalFlow)."""
    qubits: list[QubitId] = field(default_factory=list)
    """Local qubits."""

    def __post_init__(self):
        if len(self.qubits) == 0:
            self.qubits = self.action.qubits

    @property
    def id(self) -> Id:
        """Task Id."""
        return self.action.id

    @property
    def uid(self) -> TaskId:
        """Task unique Id."""
        return (self.action.id, self.iteration)

    @property
    def operation(self):
        """Routine object from Operation Enum."""
        if self.action.operation is None:
            raise RuntimeError("No operation specified")

        return Operation[self.action.operation].value

    @property
    def main(self):
        """Main node to be executed next."""
        return self.action.main

    @property
    def next(self) -> list[Id]:
        """Node unlocked after the execution of this task."""
        if self.action.next is None:
            return []
        if isinstance(self.action.next, str):
            return [self.action.next]

        return self.action.next

    @property
    def priority(self):
        """Priority level."""
        if self.action.priority is None:
            return MAX_PRIORITY
        return self.action.priority

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
        qubits: Union[Qubits, QubitsPairs] = dict,
        mode: ExecutionMode = None,
        folder: Path = None,
    ):
        completed = Completed(self, Normal(), folder)

        if mode.name in ["autocalibration", "acquire"]:
            completed.data, completed.data_time = self._acquire(
                platform=platform, qubits=qubits
            )

        if mode.name in ["autocalibration", "fit"]:
            completed.results, completed.results_time = self._fit(completed.data)

        return completed

    def _acquire(self, platform: Platform, qubits: Union[Qubits, QubitsPairs]) -> Data:
        """Acquisition

        Args:
            platform (`Platform`): Qibolab's platform
            qubits (`Union[Qubits, QubitsPairs]`): Qubit or QubitPairs dict.

        Returns:
            data (`Data`): data acquisition output
            time (float): acquisition time
        """
        try:
            operation: Routine = self.operation
            parameters = self.parameters
        except RuntimeError:
            operation = dummy_operation
            parameters = DummyPars()
        if operation.platform_dependent and operation.qubits_dependent:
            if len(self.qubits) > 0:
                if platform is not None:
                    if any(isinstance(i, tuple) for i in self.qubits):
                        qubits = allocate_qubits_pairs(platform, self.qubits)
                    else:
                        qubits = allocate_single_qubits(platform, self.qubits)
                else:
                    qubits = self.qubits

            data, time = operation.acquisition(
                parameters, platform=platform, qubits=qubits
            )
            # after acquisition we update the qubit parameter
            self.qubits = list(qubits)
        else:
            data, time = operation.acquisition(parameters, platform=platform)
        return data, time

    def _fit(self, data: Data) -> Results:
        """Fitting

        Args:
            data (`Data`): data acquisition

        Returns:
            results (`Results`): fitting output
            time (float): fitting time
        """
        operation: Routine = self.operation
        return operation.fit(data)


@dataclass
class Completed:
    """A completed task."""

    task: Task
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added

    """
    status: Status
    """Protocol status."""
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

    @property
    def datapath(self):
        """Path contaning data and results file for task."""
        path = self.folder / "data" / f"{self.task.id}_{self.task.iteration}"
        if not path.is_dir():
            path.mkdir(parents=True)
        return path

    @property
    def results(self):
        """Access task's results."""
        if not (self.datapath / RESULTSFILE).is_file():
            return None

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
        if not (self.datapath / DATAFILE).is_file():
            return None
        if self._data is None:
            Data = self.task.operation.data_type
            self._data = Data.load(self.datapath)
        return self._data

    @data.setter
    def data(self, data: Data):
        """Set and store data."""
        self._data = data
        self._data.save(self.datapath)

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)
