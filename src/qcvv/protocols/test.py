# -*- coding: utf-8 -*-
from qibo import gates, models

from qcvv.data import Data


def test(
    nqubits,
    nshots,
    points=1,
):
    data = Data("test", quantities=["samples", "nshots"])
    circuit = models.Circuit(nqubits)
    circuit.add(gates.M(0))
    execution = circuit(nshots=nshots)

    data.add({"nshots": 10, "samples": execution.probabilities()})
    yield data
