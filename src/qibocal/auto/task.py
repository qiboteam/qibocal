"""Action execution tracker."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from qibolab.platform import Platform
from qibolab.qubits import QubitId

from ..protocols.characterization import Operation
from ..utils import allocate_qubits
from .operation import (
    DATAFILE,
    RESULTFILE,
    Data,
    DummyPars,
    Qubits,
    Results,
    Routine,
    dummy_operation,
)
from .runcard import Action, Id

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed.

But not so insanely big not to fit in a native integer.

"""

TaskId = tuple[Id, int]
"""Unique identifier for executed tasks."""


@dataclass
class Task:
    action: Action
    iteration: int = 0
    qubits: List[QubitId] = field(default_factory=list)

    def __post_init__(self):
        if len(self.qubits) == 0:
            self.qubits = self.action.qubits

    @classmethod
    def load(cls, card: dict):
        descr = Action(**card)

        return cls(action=descr)

    @property
    def id(self) -> Id:
        return self.action.id

    @property
    def uid(self) -> TaskId:
        return (self.action.id, self.iteration)

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
        if self.action.next is None:
            return []
        if isinstance(self.action.next, str):
            return [self.action.next]

        return self.action.next

    @property
    def priority(self):
        if self.action.priority is None:
            return MAX_PRIORITY
        return self.action.priority

    @property
    def parameters(self):
        return self.operation.parameters_type.load(self.action.parameters)

    @property
    def update(self):
        return self.action.update

    def path(self, base_dir: Path):
        return Path(base_dir / "data" / f"{self.id}_{self.iteration}")

    def data(self, base_dir: Path = None):
        if not (self.path(base_dir) / DATAFILE).is_file():
            return None

        Data = self.operation.data_type
        return Data.load(self.path(base_dir))

    def results(self, base_dir: Path = None):
        if not (self.path(base_dir) / RESULTFILE).is_file():
            return None

        Results = self.operation.results_type
        return Results.load(self.path(base_dir))

    def run(self, folder: Path, platform: Platform, qubits: Qubits) -> Results:
        self.acquire(folder, platform, qubits)
        self.fit(folder)

    def acquire(self, folder: Path, platform: Platform, qubits: Qubits) -> Data:
        try:
            operation: Routine = self.operation
            parameters = self.parameters
        except RuntimeError:
            operation = dummy_operation
            parameters = DummyPars()
        os.makedirs(folder / "data" / f"{self.id}_{self.iteration}")
        path = self.path(folder)

        if platform is not None:
            if len(self.qubits) > 0:
                qubits = allocate_qubits(platform, self.qubits)

        data: Data = operation.acquisition(parameters, platform=platform, qubits=qubits)
        # after acquisition we update the qubit parameter
        self.qubits = list(qubits)
        data.save(path)

    def fit(self, folder: Path, **kwargs) -> Results:
        results = self.operation.fit(self.data(folder))
        results.save(self.path(folder))
