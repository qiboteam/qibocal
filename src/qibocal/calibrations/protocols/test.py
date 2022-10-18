from qibo import gates, models

from qibocal.data import Data


def test(
    platform,
    qubit: list,
    nshots,
    points=1,
):
    data = Data("test", quantities=["nshots", "probabilities"])
    nqubits = len(qubit)
    circuit = models.Circuit(nqubits)
    circuit.add(gates.H(qubit[0]))
    circuit.add(gates.H(qubit[1]))
    # circuit.add(gates.H(1))
    circuit.add(gates.M(*qubit))
    execution = circuit(nshots=nshots)

    data.add({"nshots": nshots, "probabilities": execution.probabilities()})
    yield data
