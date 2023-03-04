import copy
from dataclasses import dataclass
from typing import Dict, Tuple

from .runcard import Id
from .task import Output, Task


@dataclass
class Completed:
    task: Task
    output: Output

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)


class History(Dict[Tuple[Id, int], Completed]):
    def push(self, completed: Completed):
        self[(completed.task.id, completed.task.iteration)] = completed
        completed.task.iteration += 1

    # TODO: implemet time_travel()
