import numpy as np
from qibo import Circuit, gates
from qibolab import create_platform

from qibocal.auto.transpile import (
    dummy_transpiler,
    pad_circuit,
    set_compiler,
    transpile_circuits,
)


def test_natives():
    platform = create_platform("dummy")
    compiler = set_compiler(platform)
    transpiler = dummy_transpiler(platform)
    assert gates.iSWAP in compiler.rules

    circuit = Circuit(2)
    circuit.add(gates.iSWAP(0, 1))
    qubit_map = [1, 2]
    transpiled_circuit = transpile_circuits(
        [circuit], [qubit_map], platform, transpiler
    )[0]
    sequence, _ = compiler.compile(transpiled_circuit, platform)
    assert len(sequence) == 4  # dummy compiles iSWAP in 4 pulses


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


def test_transpile_circuits():
    platform = create_platform("dummy")
    transpiler = dummy_transpiler(platform)

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    qubit_map = [1, 2]
    transpiled_circuit = transpile_circuits(
        [circuit], [qubit_map], platform, transpiler
    )[0]

    true_circuit = Circuit(5)
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.Z(1))
    true_circuit.add(gates.Z(2))
    assert np.all(true_circuit.unitary() == transpiled_circuit.unitary())
