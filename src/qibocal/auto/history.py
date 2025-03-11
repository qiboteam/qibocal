"""Track execution history."""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterator, Optional

from .task import Completed, Id, TaskId

HISTORY = "history.json"
"""File where protocols order is dumped."""


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

    @property
    def _serialized_order(self) -> list[str]:
        """JSON friendly _order attribute.

        _order, which is a list of TaskId objects, is not JSON serializable,
        it is converted into a list of strings which match the data folder for each
        protocol.
        """
        return [str(i) for i in self._order]

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
    def load(cls, path: Path) -> "History":
        """Load history from path.

        Fill history with protocols contained in `data` path. If `history.json` is present,
        they are sorted based on that, otherwise they are sorted by modification time.
        """
        instance = cls()
        if not (path / HISTORY).exists():
            protocols = sorted((path / "data").glob("*"), key=os.path.getmtime)
        else:
            raw_protocols = json.loads((path / HISTORY).read_text())
            protocols = [path / "data" / protocol for protocol in raw_protocols]

        for protocol in protocols:
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
    def route(task_id: TaskId, folder: Path) -> Path:
        """Determine the path related to a completed task given TaskId.

        `folder` should be usually the general output folder, used by Qibocal to store
        all the execution results. Cf. :class:`qibocal.auto.output.Output`.
        """
        return folder / "data" / f"{task_id}"

    def flush(self, output: Optional[Path] = None):
        """Flush all content to disk.

        Specifying `output` is possible to select which folder should be considered as
        the general Qibocal output folder. Cf. :class:`qibocal.auto.output.Output`.
        """
        for task_id, completed in self.items():
            if output is not None:
                completed.path = self.route(task_id, output)
            completed.flush()
        (output / HISTORY).write_text(json.dumps(self._serialized_order, indent=4))

    # TODO: implement time_travel()


class DummyHistory:
    """Empty History object, used by `qq compare`.

    Used for comparing multiple reports, as their history is not saved.
    """

    def flush(self, output=None):
        pass

    def items(self):
        return tuple()
