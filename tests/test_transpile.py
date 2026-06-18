from collections import Counter

import numpy as np
import pytest
from qibo import Circuit, gates
from qibolab import (
    Acquisition,
    AcquisitionType,
    AveragingMode,
    PulseSequence,
    create_platform,
)

from qibocal.auto.operation import QubitId
from qibocal.auto.transpile import (
    _execute_circuits,
    _pad_circuit,
    _string_to_integer_qubit_map,
    _transpile_circuit,
    _validate_measurement,
    build_native_gate_compiler,
    build_native_gate_transpiler,
    execute_circuits,
)


def test_natives():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    transpiler = build_native_gate_transpiler(platform)
    assert gates.iSWAP in compiler.rules

    circuit = Circuit(2)
    circuit.add(gates.iSWAP(0, 1))
    qubit_map: list[int | str] = [1, 2]
    transpiled_circuit = _transpile_circuit(circuit, qubit_map, platform, transpiler)
    sequence, _ = compiler.compile(transpiled_circuit, platform)
    assert len(sequence) == 4  # dummy compiles iSWAP in 4 pulses


def test_pad_circuit():
    small_circuit = Circuit(2)
    small_circuit.add(gates.X(0))
    small_circuit.add(gates.X(1))
    qubit_map = [1, 2]
    big_circuit = _pad_circuit(4, small_circuit, qubit_map)

    true_circ = Circuit(4)
    true_circ.add(gates.X(1))
    true_circ.add(gates.X(2))
    assert np.all(true_circ.unitary() == big_circuit.unitary())


def test_transpile_circuits():
    platform = create_platform("dummy")
    transpiler = build_native_gate_transpiler(platform)

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    qubit_map: list[QubitId] = [1, 2]
    transpiled_circuit = _transpile_circuit(circuit, qubit_map, platform, transpiler)

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

    qubit_map = _string_to_integer_qubit_map(["q2", "q0"], PlatformStub())

    transpiled_circuit = _transpile_circuit(
        circuit, qubit_map, PlatformStub(), mock_transpiler
    )

    expected = Circuit(3)
    expected.add(gates.X(2))
    expected.add(gates.X(0))

    assert np.all(expected.unitary() == transpiled_circuit.unitary())


def test_execute_circuits_qubit_mapping():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    transpiler = build_native_gate_transpiler(platform)

    circuit = Circuit(2)
    circuit.add(gates.M(*range(2)))

    qubit_map = [0, 1]
    qubit_maps = [qubit_map] * 2
    circuits = [circuit] * 2

    # No qubit mapping passed
    with pytest.raises(AssertionError):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            transpiler=transpiler,
            circuits=[circuit],
            nshots=20,
        )

    # Number of qubit mapping not matching number of circuits
    with pytest.raises(AssertionError):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=circuits,
            transpiler=transpiler,
            nshots=20,
            qubit_maps=[qubit_map],
        )
    with pytest.raises(AssertionError):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            transpiler=transpiler,
            circuits=[circuit],
            nshots=20,
            qubit_maps=qubit_maps,
        )

    # qubit_maps and qubit_map being passed to exec call
    with pytest.raises(AssertionError):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=circuits,
            transpiler=transpiler,
            nshots=20,
            qubit_maps=qubit_maps,
            qubit_map=qubit_maps,
        )

    execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=circuits,
        transpiler=transpiler,
        nshots=20,
        qubit_maps=qubit_maps,
    )


def test_measurement_validation():
    meas = gates.M(0)
    acq = Acquisition(duration=20)
    seq = PulseSequence([("0/acq", acq)])
    readout = {acq.id: np.zeros(20)}

    with pytest.raises(
        KeyError, match=f"Acquisition ID {acq.id} not found in readout results."
    ):
        _validate_measurement(meas, seq, {})

    with pytest.raises(AssertionError):
        _validate_measurement(gates.M(*range(2)), seq, readout)


def test_execute_circuits_single_shot():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    pair = (0, 1)
    circuit = Circuit(2)
    circuit.add(gates.M(*pair))
    nshots = 32

    [results] = _execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit],
        nshots=nshots,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    [counts] = results[pair]
    assert sum(counts.values()) == nshots
    assert set(counts).issubset({"00", "01", "10", "11"})


def test_execute_circuits_cyclic():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    circuit = Circuit(2)
    qubit = 0
    circuit.add(gates.M(qubit))
    nshots = 20

    [results] = _execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit],
        nshots=nshots,
        averaging_mode=AveragingMode.CYCLIC,
    )

    [counts] = results[qubit]
    assert set(counts) == {"0", "1"}
    assert sum(counts.values()) == nshots


def test_execute_circuits_cyclic_raises_for_multi_qubit():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    circuit = Circuit(2)
    circuit.add(gates.M(*range(2)))

    with pytest.raises(
        ValueError,
        match="Hardware averaging is only supported for single qubit readout.",
    ):
        _execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=[circuit],
            nshots=20,
            averaging_mode=AveragingMode.CYCLIC,
        )


def test_execute_circuits_cyclic_maps_readout_to_circuit_order(monkeypatch):
    # Test that we correctly use the acquisition ids to associate the results with the
    # sequence in execute_circuits. This way we don't depend on the order in which the
    # platform returns the results, which may not always remain the same as the sequence
    # order, e.g. when batching reorders to optimize resource in hardware
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    nshots = 20

    circuit0 = Circuit(1)
    circuit0.add(gates.M(0))
    circuit1 = Circuit(1)
    circuit1.add(gates.M(0))

    qubit = list(platform.qubits)[0]

    def execute_reversed_order(sequences, averaging_mode, acquisition_type, **options):
        assert averaging_mode == AveragingMode.CYCLIC
        assert acquisition_type == AcquisitionType.DISCRIMINATION
        first_id = sequences[0].acquisitions[0][1].id
        second_id = sequences[1].acquisitions[0][1].id
        return {
            second_id: 0.8,
            first_id: 0.3,
        }

    monkeypatch.setattr(platform, "execute", execute_reversed_order)

    results = _execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=[circuit0, circuit1],
        nshots=nshots,
        averaging_mode=AveragingMode.CYCLIC,
    )

    countslist = [counts for res in results for counts in res[qubit]]

    assert countslist == [Counter({"0": 14, "1": 6}), Counter({"0": 4, "1": 16})]
