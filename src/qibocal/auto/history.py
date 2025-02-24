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

    def _pending_task_id(self, _id: Id) -> TaskId:
        """Retrieve the TaskId of a given task to be executed."""
        return TaskId(id=_id, iteration=len(self._tasks[_id]))

    def _executed_task_id(self, _id: Id) -> TaskId:
        """Retrieve the TaskId of a given executed task."""
        return TaskId(id=_id, iteration=len(self._tasks[_id]) - 1)

    def push(self, completed: Completed) -> TaskId:
        """Adding completed task to history."""
        id = completed.task.id
        self._tasks[id].append(completed)
        task_id = self._executed_task_id(id)
        self._order.append(task_id)
        return task_id

    def task_path(self, task_id: TaskId, folder: Optional[Path]) -> Path:
        """Determine the path related to a completed task given TaskId.

        `folder` should be usually the general output folder, used by Qibocal to store
        all the execution results. Cf. :class:`qibocal.auto.output.Output`.
        """
        if folder is None:
            return None
        return folder / "data" / f"{task_id}"

    # TODO: implement time_travel()


class DummyHistory:
    """Empty History object, used by `qq compare`.

    Used for comparing multiple reports, as their history is not saved.
    """

    def items(self):
        return tuple()
