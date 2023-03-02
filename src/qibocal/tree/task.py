from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from operation import Operation


class Parameters(ABC):
    @classmethod
    def load(cls):
        pass


class Update(ABC):
    pass


@dataclass
class Output:  # TODO: write Output as abstract class
    results: str
    update: Update


class Keywords(Enum):
    id = "id"
    operation = "operation"
    main = "main"
    next = "next"
    priority = "priority"


@dataclass
class Task:
    id: str
    operation: Operation
    parameters: Parameters
    output: Optional[Output] = None

    @classmethod
    def load(cls, card: Dict[str, Union[str, int, float, list]]):
        operation = card[Keywords.operation.value]
        assert isinstance(operation, str)

        id_ = card[Keywords.id.value]
        assert isinstance(id_, str)

        parameters = {}
        for name, value in card:
            if name in Keywords:
                continue

            parameters[name] = value

        return cls(
            id=id_,
            operation=Operation[operation],
            parameters=Parameters.load(parameters),
        )

    def run(self) -> Output:
        self.output = Output(self.operation.value.routine(self.parameters))
        return Output

    def complete(self, completed_id):
        # This function takes the ID of a completed Task
        # and updates the requirements

        if completed_id in self._requirements.keys():
            self._requirements[completed_id] = True
