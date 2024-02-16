"""Action execution tracker."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mode
from typing import Optional, Union

from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibolab.serialize import dump_platform

from ..config import log, raise_error
from ..protocols.characterization import Operation
from ..utils import (
    allocate_qubits_pairs,
    allocate_single_qubits,
    allocate_single_qubits_lists,
)
from .mode import ExecutionMode
from .operation import (
    Data,
    DummyPars,
    Qubits,
    QubitsPairs,
    Results,
    Routine,
    dummy_operation,
)
from .runcard import Action, Id
from .status import Failure, Normal

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed. But not so insanely big not to fit in a native integer."""
TaskId = tuple[Id, int]
"""Unique identifier for executed tasks."""
PLATFORM_DIR = "platform"
"""Folder where platform will be dumped."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    iteration: int = 0
    """Task iteration."""
    qubits: list[QubitId, QubitPairId] = field(default_factory=list)
    """List of QubitIds or QubitPairIds for task."""

    def __post_init__(self):
        if len(self.qubits) == 0:
            self.qubits = self.action.qubits

    # TODO: to be removed in https://github.com/qiboteam/qibocal/pull/682
    def _allocate_local_qubits(self, qubits, platform):
        """Create qubits dictionary from QubitIds."""
        if len(self.qubits) > 0:
            if platform is not None:
                if any(isinstance(i, tuple) for i in self.qubits):
                    task_qubits = allocate_qubits_pairs(platform, self.qubits)
                elif any(
                    isinstance(i, tuple) or isinstance(i, list) for i in self.qubits
                ):
                    task_qubits = allocate_single_qubits_lists(platform, self.qubits)
                else:
                    task_qubits = allocate_single_qubits(platform, self.qubits)
            else:
                # None platform use just ids
                task_qubits = self.qubits
        else:
            task_qubits = qubits

        self.qubits = list(task_qubits)

        return task_qubits

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
        max_iterations: int,
        platform: Platform = None,
        qubits: Union[Qubits, QubitsPairs] = dict,
        mode: ExecutionMode = None,
        folder: Path = None,
    ):
        if self.iteration > max_iterations:
            raise_error(
                ValueError,
                f"Maximum number of iterations {max_iterations} reached!",
            )

        completed = Completed(self, folder)

        try:
            if self.parameters.nshots is None:
                self.action.parameters["nshots"] = platform.settings.nshots
            if self.parameters.relaxation_time is None:
                self.action.parameters["relaxation_time"] = (
                    platform.settings.relaxation_time
                )
            operation: Routine = self.operation
            parameters = self.parameters

        except (RuntimeError, AttributeError):
            operation = dummy_operation
            parameters = DummyPars()
        if mode.name in ["autocalibration", "acquire"]:
            if operation.platform_dependent and operation.qubits_dependent:
                completed.data, completed.data_time = operation.acquisition(
                    parameters,
                    platform=platform,
                    qubits=self._allocate_local_qubits(qubits, platform),
                )
                # need to reassign qubit since when task
                # is passed it is deepcopied
                completed.task.qubits = self.qubits
            else:
                completed.data, completed.data_time = operation.acquisition(
                    parameters, platform=platform
                )
        if mode.name in ["autocalibration", "fit"]:
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
        path = self.folder / "data" / f"{self.task.id}_{self.task.iteration}"
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

    def update_platform(self, platform: Platform, update: bool):
        """Perform update on platform' parameters by looping over qubits or pairs."""
        if self.task.update and update:
            for qubit in self.task.qubits:
                try:
                    self.task.operation.update(self.results, platform, qubit)
                except KeyError:
                    log.warning(
                        f"Skipping update of qubit {qubit} due to error in fit."
                    )
            (self.datapath / PLATFORM_DIR).mkdir(parents=True, exist_ok=True)
            dump_platform(platform, self.datapath / PLATFORM_DIR)

    def validate(self) -> tuple[Optional[TaskId], Optional[dict]]:
        """Check status of completed and handle Failure using handler."""
        if self.task.action.validator is not None:
            status = []
            for target in self.task.qubits:
                # TODO: how to handle multiple targets?
                # dummy solution for now: take the mode.
                qubit_status, params = self.task.action.validator.validate(
                    self.results, target
                )
                status.append(qubit_status)
            output = mode(status)
            if isinstance(output, Failure):
                return None, None
            elif isinstance(output, Normal):
                return self.task.id, None
            else:
                return output, params

        return self.task.id, None
