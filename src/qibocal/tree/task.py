from abc import ABC
from dataclasses import dataclass

from operation import *


class Parameters(ABC):
    pass


class Update(ABC):
    pass


@dataclass
class Output(ABC):
    pass


@dataclass
class Task:
    operation: Operation
    parameters: Parameters
    _requirements: dict[str, bool]

    @property
    def ready(self):
        return all(self.requirements.values())

    @property
    def requirements(self):
        return self._requirements

    @classmethod
    def load(cls, card: list):
        name = card[0]
        parameters = card[1]
        if card[2][0] == "start":
            requirements = {"start": True}
        else:
            requirements = {i: False for i in card[2]}
        ready = all(requirements.values())
        return cls(
            operation=Operation[name], parameters=parameters, _requirements=requirements
        )

    def run(self) -> Output:
        self.operation.value.routine(self.parameters)
