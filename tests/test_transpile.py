import numpy as np
from qibo import Circuit, gates
from qibo.backends import construct_backend

from qibocal.auto.transpile import (
    dummy_transpiler,
    execute_transpiled_circuit,
    execute_transpiled_circuits,
    pad_circuit,
)


def test_padd_circuit():
    small_circuit = Circuit(2)
    small_circuit.add(gates.X(0))
    small_circuit.add(gates.X(1))
    qubit_map = [1, 2]
    big_circuit = pad_circuit(4, small_circuit, qubit_map)

    true_circ = Circuit(4)
    true_circ.add(gates.X(1))
    true_circ.add(gates.X(2))
    assert np.all(true_circ.unitary() == big_circuit.unitary())


def test_execute_transpiled_circuit():

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    qubit_map = [1, 2]
    backend = construct_backend("qibolab", platform="dummy")
    transpiler = dummy_transpiler(backend)
    transpiled_circuit, _ = execute_transpiled_circuit(
        circuit, qubit_map, backend, transpiler=transpiler
    )
    true_circuit = Circuit(5)
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.Z(1))
    true_circuit.add(gates.Z(2))
    assert np.all(true_circuit.unitary() == transpiled_circuit.unitary())


def test_execute_transpiled_circuits():

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    qubit_map = [1, 2]
    backend = construct_backend("qibolab", platform="dummy")
    transpiler = dummy_transpiler(backend)
    transpiled_circuits, _ = execute_transpiled_circuits(
        [circuit], [qubit_map], backend, transpiler=transpiler
    )
    true_circuit = Circuit(5)
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.Z(1))
    true_circuit.add(gates.Z(2))
    assert np.all(true_circuit.unitary() == transpiled_circuits[0].unitary())
