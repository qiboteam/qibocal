from abc import ABC
from dataclasses import dataclass
from typing import Optional

from .operation import Operation
from .runcard import Action


class Parameters(ABC):
    @classmethod
    def load(cls, parameters):
        return cls()


class Update(ABC):
    pass


@dataclass
class Output:  # TODO: write Output as abstract class
    results: str
    update: Update


@dataclass
class Task:
    id: str
    operation: Operation
    parameters: Parameters
    output: Optional[Output] = None

    @classmethod
    def load(cls, card: dict):
        descr = Action(**card)

        return cls(
            id=descr.id,
            #  operation=Operation[descr.id],
            operation=Operation.command_1,
            parameters=Parameters.load(descr.pars),
        )

    def run(self):
        self.output = Output(self.operation.value.routine(self.parameters), Update())

    def complete(self, completed_id):
        pass
