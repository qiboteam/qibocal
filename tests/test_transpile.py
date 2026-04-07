from collections import Counter

import numpy as np
import pytest
from qibo import Circuit, gates
from qibolab import AveragingMode, create_platform

from qibocal.auto.operation import QubitId
from qibocal.auto.transpile import (
    dummy_transpiler,
    execute_circuits,
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
    qubit_map: list[int | str] = [1, 2]
    [transpiled_circuit] = transpile_circuits(
        [circuit], [qubit_map], platform, transpiler
    )
    sequence, _ = compiler.compile(transpiled_circuit, platform)
    assert len(sequence) == 4  # dummy compiles iSWAP in 4 pulses


def test_pad_circuit():
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
    qubit_map: list[QubitId] = [1, 2]
    [transpiled_circuit] = transpile_circuits(
        [circuit], [qubit_map], platform, transpiler
    )

    true_circuit = Circuit(5)
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(1, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.GPI2(2, np.pi / 2))
    true_circuit.add(gates.Z(1))
    true_circuit.add(gates.Z(2))
    assert np.all(true_circuit.unitary() == transpiled_circuit.unitary())


def test_transpile_circuits_with_string_qubit_ids():
    class PlatformStub:
        qubits = ["q0", "q1", "q2"]
        nqubits = 3

    def mock_transpiler(circuit):
        "Mock a call to the Passes transpiler."
        return (circuit, None)

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))

    [transpiled_circuit] = transpile_circuits(
        [circuit], [["q2", "q0"]], PlatformStub(), mock_transpiler
    )

    expected = Circuit(3)
    expected.add(gates.X(2))
    expected.add(gates.X(0))

    assert np.all(expected.unitary() == transpiled_circuit.unitary())


def test_execute_circuits_single_shot():
    platform = create_platform("dummy")
    compiler = set_compiler(platform)
    circuit = Circuit(2)
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    qubit_map = list(platform.qubits)[:2]
    nshots = 32

    [counts] = execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit],
        qubit_maps=[qubit_map],
        nshots=nshots,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    assert sum(counts.values()) == nshots
    assert set(counts).issubset({"00", "01", "10", "11"})


def test_execute_circuits_cyclic():
    platform = create_platform("dummy")
    compiler = set_compiler(platform)
    circuit = Circuit(2)
    circuit.add(gates.M(0))
    qubit_map = [list(platform.qubits)[0]]
    nshots = 20

    [counts] = execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit],
        qubit_maps=[qubit_map],
        nshots=nshots,
        averaging_mode=AveragingMode.CYCLIC,
    )

    assert set(counts) == {"0", "1"}
    assert sum(counts.values()) == nshots


def test_execute_circuits_cyclic_raises_for_multi_qubit():
    platform = create_platform("dummy")
    compiler = set_compiler(platform)
    circuit = Circuit(2)
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    qubit_map = list(platform.qubits)[:2]

    with pytest.raises(ValueError, match="CYCLIC only supports single qubit readout"):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=[circuit],
            qubit_maps=[qubit_map],
            nshots=20,
            averaging_mode=AveragingMode.CYCLIC,
        )


def test_execute_circuits_cyclic_maps_readout_to_circuit_order(monkeypatch):
    # Test that we correctly use the acquisition ids to associate the results with the
    # sequence in execute_circuits. This way we don't depend on the order in which the
    # platform returns the results, which may not always remain the same as the sequence
    # order, e.g. when batching reorders to optimize resource in hardware
    platform = create_platform("dummy")
    compiler = set_compiler(platform)
    nshots = 20

    circuit0 = Circuit(1)
    circuit0.add(gates.M(0))
    circuit1 = Circuit(1)
    circuit1.add(gates.M(0))

    qubit = list(platform.qubits)[0]
    qubit_maps = [[qubit], [qubit]]

    def execute_reversed_order(sequences, nshots, averaging_mode):
        assert averaging_mode == AveragingMode.CYCLIC
        first_id = sequences[0].acquisitions[0][1].id
        second_id = sequences[1].acquisitions[0][1].id
        return {
            second_id: 0.8,
            first_id: 0.3,
        }

    monkeypatch.setattr(platform, "execute", execute_reversed_order)

    countslist = execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit0, circuit1],
        qubit_maps=qubit_maps,
        nshots=nshots,
        averaging_mode=AveragingMode.CYCLIC,
    )

    assert countslist == [Counter({"0": 14, "1": 6}), Counter({"0": 4, "1": 16})]
