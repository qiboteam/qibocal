from dataclasses import dataclass
from typing import List

from .task import Output, Task


@dataclass
class Completed:
    task: Task
    output: Output


class History(List[Completed]):
    pass
    # TODO: implemet time_travel()
