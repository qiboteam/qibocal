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

from qibocal.auto.transpile import (
    _execute_circuits,
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

    circuit = Circuit(2, wire_names=[1, 2])
    circuit.add(gates.iSWAP(0, 1))
    transpiled_circuit, _ = transpiler(circuit)
    sequence, _ = compiler.compile(transpiled_circuit, platform)
    assert len(sequence) == 4  # dummy compiles iSWAP in 4 pulses


def test_execute_circuits_qubit_mapping():
    platform = create_platform("dummy")
    compiler = build_native_gate_compiler(platform)
    transpiler = build_native_gate_transpiler(platform)

    circuit = Circuit(2)
    circuit.add(gates.M(*range(2)))

    qubit_map = [0, 1]
    qubit_maps = [qubit_map] * 2
    circuits = [circuit] * 2

    with pytest.raises(AssertionError):
        execute_circuits(
            platform=platform,
            compiler=compiler,
            transpiler=transpiler,
            circuits=[circuit],
            nshots=20,
            qubit_maps=qubit_maps,
        )

    execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=circuits,
        transpiler=transpiler,
        nshots=20,
        qubit_maps=qubit_maps,
    )

    execute_circuits(
        platform=platform,
        compiler=compiler,
        circuits=circuits,
        transpiler=transpiler,
        nshots=20,
        qubit_maps=[qubit_map],
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
