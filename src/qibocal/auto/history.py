"""Track execution history."""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterator, Optional

from .task import Completed, Id, TaskId


@dataclass
class History:
    """Execution history.

    This is not only used for logging and debugging, but it is an actual
    part of the execution itself, since later routines can retrieve the
    output of former ones from here.
    """

    _tasks: dict[Id, list[Completed]] = field(default_factory=lambda: defaultdict(list))
    """List of completed tasks.

    .. note::

        Representing the object as a map of sequences makes it smoother to identify the
        iterations of a given task, since they are already grouped together.
    """
    _order: list[TaskId] = field(default_factory=list)
    """Record of the execution order."""

    @singledispatchmethod
    def __contains__(self, elem: Id):
        """Check whether a generic or specific task has been completed."""
        return elem in self._tasks

    @__contains__.register
    def _(self, elem: TaskId):
        return len(self._tasks.get(elem.id, [])) > elem.iteration

    @singledispatchmethod
    def __getitem__(self, _):
        """Access a generic or specific task."""
        raise NotImplementedError

    @__getitem__.register(str)
    def _(self, key: Id) -> list[Completed]:
        return self._tasks[key]

    @__getitem__.register
    def _(self, key: TaskId) -> Completed:
        return self._tasks[key.id][key.iteration]

    def __iter__(self) -> Iterator[TaskId]:
        """Iterate individual tasks identifiers.

        It follows the execution order.
        """
        return iter(self._order)

    def values(self) -> Iterator[Completed]:
        """Iterate individual tasks according to the execution order."""
        return (self[task_id] for task_id in self)

    def items(self) -> Iterator[tuple[TaskId, Completed]]:
        """Consistent iteration over individual tasks and their ids."""
        return ((task_id, self[task_id]) for task_id in self)

    @classmethod
    def load(cls, path: Path):
        """To be defined."""
        instance = cls()
        for protocol in (path / "data").glob("*"):
            instance.push(Completed.load(protocol))
        return instance

    def push(self, completed: Completed) -> TaskId:
        """Adding completed task to history."""
        id = completed.task.id
        self._tasks[id].append(completed)
        task_id = TaskId(id=id, iteration=len(self._tasks[id]) - 1)
        self._order.append(task_id)
        return task_id

    @staticmethod
    def route(completed: Completed, folder: Path):
        """Determine the path related to a completed task.

        `folder` should be ussually the general output folder, used by Qibocal to store
        all the execution results. Cf. :cls:`qibocal.auto.output.Output`.
        """
        return folder / "data" / f"{completed.task.id}"

    def flush(self, output: Optional[Path] = None):
        """Flush all content to disk.

        Specifying `output` is possible to select which folder should be considered as
        the general Qibocal output folder. Cf. :cls:`qibocal.auto.output.Output`.
        """
        for task in self._tasks.values():
            for completed in task:
                if output is not None:
                    completed.path = self.route(completed, output)
                completed.dump()

    # TODO: implement time_travel()
