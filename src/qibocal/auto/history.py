"""Track execution history."""
import copy
from dataclasses import dataclass
from typing import Dict, Tuple

from .operation import Data, Results
from .runcard import Id
from .status import Status
from .task import Task


@dataclass
class Completed:
    """A completed task."""

    task: Task
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added

    """
    data: Data
    """Data acquired."""
    results: Results
    """Final results."""
    status: Status
    """Routine status."""

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)


class History(Dict[Tuple[Id, int], Completed]):
    """Execution history.

    This is not only used for logging and debugging, but it is an actual part
    of the execution itself, since later routines can retrieve the output of
    former ones from here.

    """

    def push(self, completed: Completed):
        self[(completed.task.id, completed.task.iteration)] = completed
        completed.task.iteration += 1

    # TODO: implemet time_travel()
