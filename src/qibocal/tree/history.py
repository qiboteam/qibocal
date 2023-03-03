from dataclasses import dataclass
from typing import Dict, Tuple


from .runcard import Id
from .task import Output, Task


@dataclass
class Completed:
    task: Task
    output: Output


class History(Dict[Tuple[Id, int], Completed]):
    pass
    # TODO: implemet time_travel()
