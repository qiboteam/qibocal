from abc import ABC
from dataclasses import dataclass

from operation import Operation


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
    requirements: dict

    @classmethod
    def load(cls, card):
        name = ...
        parameters = ...
        requirements = ...
        return cls(
            operation=Operation[name], parameters=parameters, requirements=requirements
        )

    def run(self) -> Output:
        self.operation.value(self.parameters)
