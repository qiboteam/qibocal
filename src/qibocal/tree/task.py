from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .operation import Operation


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
class Description:
    id: str
    operation: str
    main: Optional[str] = None
    next: Optional[List[str]] = None
    priority: Optional[int] = None
    pars: Optional[Dict[str, Any]] = None


@dataclass
class Task:
    id: str
    operation: Operation
    parameters: Parameters
    output: Optional[Output] = None

    @classmethod
    def load(cls, card: dict):
        descr = Description(**card)

        return cls(
            id=descr.id,
            operation=Operation[descr.id],
            parameters=Parameters.load(descr.pars),
        )

    def run(self):
        self.output = Output(self.operation.value.routine(self.parameters), Update())

    def complete(self, completed_id):
        pass
