from abc import ABC
from dataclasses import dataclass

from operation import *


class Parameters(ABC):
    pass


class Update(ABC):
    pass


@dataclass
class Output:  # TODO: write Output as abstract class
    # status: Status
    results: str
    # update: Update


@dataclass
class Task:
    id: str
    operation: Operation
    parameters: Parameters
    _requirements: dict[str, bool]
    output: Output = Output("")

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

        return cls(
            id=name,
            operation=Operation[name],
            parameters=parameters,
            _requirements=requirements,
        )

    def run(self) -> Output:
        self.output = Output(self.operation.value.routine(self.parameters))
        return Output

    def complete(self, completed_id):
        # This function takes the ID of a completed Task
        # and updates the requirements

        if completed_id in self._requirements.keys():
            self._requirements[completed_id] = True
