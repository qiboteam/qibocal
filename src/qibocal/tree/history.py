from dataclasses import dataclass
from typing import List

from .task import Task, Output


@dataclass
class Completed:
    task: Task
    output: Output


@dataclass
class History:
    steps: List[Completed]

    def record(self, task: Task, output: Output):
        completed = Completed(task, output)
        self.steps.append(completed)

    # TODO: implemet time_travel()
