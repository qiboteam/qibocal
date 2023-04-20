"""Action execution tracker."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from qibolab.platforms.abstract import AbstractPlatform

from qibocal.protocols.characterization import Operation

from .operation import Data, DummyPars, Qubits, Results, Routine, dummy_operation
from .runcard import Action, Id

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed.

But not so insanely big not to fit in a native integer.

"""

DATAFILE = "data.csv"
"""Name of the file where data acquired by calibration are dumped."""


@dataclass
class Task:
    action: Action
    iteration: int = 0

    @classmethod
    def load(cls, card: dict):
        descr = Action(**card)

        return cls(action=descr)

    @property
    def id(self):
        return self.action.id

    @property
    def uid(self):
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
    def data_acquired(self):
        return self._data_acquired

    def datapath(self, base_dir: Path):
        path = base_dir / "data" / f"{self.id}_{self.iteration}"
        os.makedirs(path)
        return path

    def data(self, base_dir: Path) -> Optional[Data]:
        if not self.datapath(base_dir).is_file():
            return None
        Data = self.operation.data_type
        # print(self.datapath(Path(base_dir)))
        return Data.load_data(
            base_dir, "data", f"{self.id}_{self.iteration}", "csv", "data"
        )
        # return Data(yaml.safe_load(self.datapath(base_dir).read_text(encoding="utf-8")))

    def run(self, platform: AbstractPlatform, qubits: Qubits, folder: Path) -> Results:
        try:
            operation: Routine = self.operation
            parameters = self.parameters
        except RuntimeError:
            operation = dummy_operation
            parameters = DummyPars()

        path = self.datapath(folder)
        # TODO: fix data attributes
        self._data_acquired: Data = operation.acquisition(platform, qubits, parameters)
        self._data_acquired.to_csv(path)
        # TODO: data dump
        # path.write_text(yaml.dump(pydantic_encoder(self.data(base_dir))))
        return operation.fit(self._data_acquired)
