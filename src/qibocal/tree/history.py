from dataclasses import dataclass

from task import *


@dataclass
class Completed:
    task: Task
    output: Output


@dataclass
class History:
    steps: list[Completed]

    def record(self, task: Task, output: Output):
        completed = Completed(task, output)
        self.steps.append(completed)

    # TODO: implemet time_travel()
