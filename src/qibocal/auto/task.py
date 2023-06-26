"""Action execution tracker."""
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from qibo.config import log
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from ..protocols.characterization import Operation
from ..utils import allocate_qubits
from .operation import Data, DummyPars, Qubits, Results, Routine, dummy_operation
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
    qubits: list[QubitId] = field(default_factory=list)

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
    def next(self) -> list[Id]:
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

    @property
    def data(self):
        return self._data

    def datapath(self, base_dir: Path):
        path = base_dir / "data" / f"{self.id}_{self.iteration}"
        os.makedirs(path)
        return path

    def run(self, folder: Path, platform: Platform, qubits: Qubits) -> Results:
        try:
            operation: Routine = self.operation
            parameters = self.parameters
        except RuntimeError:
            operation = dummy_operation
            parameters = DummyPars()

        path = self.datapath(folder)
        if operation.platform_dependent and operation.qubits_dependent:
            if len(self.qubits) > 0:
                if platform is not None:
                    qubits = allocate_qubits(platform, self.qubits)
                else:
                    qubits = self.qubits

            start_acq = time.time()
            self._data: Data = operation.acquisition(
                parameters, platform=platform, qubits=qubits
            )
            stop_acq = time.time()
            # after acquisition we update the qubit parameter
            self.qubits = list(qubits)

        else:
            self._data: Data = operation.acquisition(parameters, platform=platform)
        self._data.save(path)

        start_fit = time.time()
        res = operation.fit(self._data)
        stop_fit = time.time()

        log.info(f"Acquisition time: {stop_acq - start_acq}")
        log.info(f"Fit time: {stop_fit - start_fit}")

        return res
