from abc import ABC
from dataclasses import dataclass
from typing import List

from .operation import Operation
from .runcard import Action, Id


class Parameters(ABC):
    @classmethod
    def load(cls, parameters):
        return cls()


class Update(ABC):
    pass


@dataclass
class Output:
    # TODO: write Output as abstract class
    results: str
    update: Update


@dataclass
class Task:
    action: Action
    time: int = 0

    @classmethod
    def load(cls, card: dict):
        descr = Action(**card)

        return cls(action=descr)

    @property
    def id(self):
        return self.action.id

    @property
    def operation(self):
        if self.action.operation is None:
            raise RuntimeError("No operation specified")

        return Operation[self.action.operation].value

    @property
    def main(self):
        return self.action.main

    @property
    def next(self) -> List[Id]:
        assert self.action.next is not None
        assert not isinstance(self.action.next, str)
        return self.action.next

    @property
    def priority(self):
        return self.action.priority

    @property
    def parameters(self):
        return Parameters.load(self.action.parameters)

    def run(self):
        #  return Output(self.operation.routine(self.parameters), Update())
        return Output("", Update())
