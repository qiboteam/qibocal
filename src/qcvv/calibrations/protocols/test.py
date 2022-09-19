# -*- coding: utf-8 -*-
from qibo import gates, models

from qcvv.data import Data


def test(
    platform,
    qubit,
    nshots,
    points=1,
):
    data = Data("test", quantities=["nshots"])
    circuit = models.Circuit(1)
    circuit.add(gates.H(0))
    # circuit.add(gates.H(1))
    circuit.add(gates.M(0))
    execution = circuit(nshots=nshots)

    data.add({"nshots": nshots})
    yield data
