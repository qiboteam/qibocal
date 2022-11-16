from __future__ import annotations

import pickle
from collections.abc import Iterable
from os.path import isfile

import numpy as np
from qibo import gates
from qibo.models import Circuit

from qibocal.calibrations.protocols.abstract import Experiment, SingleCliffordsFactory
from qibocal.calibrations.protocols.utils import (
    ONEQUBIT_CLIFFORD_PARAMS,
    experiment_directory,
)


class SingleCliffordsInvFactory(SingleCliffordsFactory):
    def __init__(self, qubits: list, sequences: list, runs: int) -> None:
        super().__init__(qubits, sequences, runs)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gate())
        # Build a gate out of the unitary of the whole circuit and
        # take the daggered version of that.
        circuit.add(gates.Unitary(circuit.unitary(), *self.qubits).dagger())
        circuit.add(gates.M(*self.qubits))
        return circuit


class StandardRbExperiment(Experiment):
    def __init__(
        self, circuitfactory: Iterable, nshots: int = None, data: list = None
    ) -> None:
        super().__init__(circuitfactory, nshots, data)

    def single_task(self, circuit: Circuit, datarow: dict) -> None:
        """Executes a circuit, returns the single shot results
        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        data = super().single_task(circuit, datarow)
        # Substract 1 for sequence length to not count the inverse gate.
        data["depth"] = len(circuit.queue) - 1
        return
