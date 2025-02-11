from typing import Optional

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId


def transpile_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    backend: Backend,
    transpiler: Optional[Passes],
):
    """Transpile and pad `circuits` according to the platform.

    Apply the `transpiler` to `circuits` and pad them in
    circuits with the same number of qubits in the platform.
    Before manipulating the circuits, this function check that the
    `qubit_maps` contain string ids and in the positive case it
    remap them in integers, following the ids order provided by the
    platform.

    .. note::

        In this function we are implicitly assume that the qubit ids
        are all string or all integers.
    """
    transpiled_circuits = []
    platform = backend.platform
    qubits = list(platform.qubits)
    if isinstance(qubit_maps[0][0], str):
        for i, qubit_map in enumerate(qubit_maps):
            qubit_map = map(lambda x: qubits.index(x), qubit_map)
            qubit_maps[i] = list(qubit_map)
    platform_nqubits = platform.nqubits
    for circuit, qubit_map in zip(circuits, qubit_maps):
        new_circuit = pad_circuit(platform_nqubits, circuit, qubit_map)
        transpiled_circ, _ = transpiler(new_circuit)
        transpiled_circuits.append(transpiled_circ)

    return transpiled_circuits


def execute_transpiled_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    backend: Backend,
    transpiler: Optional[Passes],
    initial_states=None,
    nshots=1000,
):
    """Transpile `circuits`.

    If the `qibolab` backend is used, this function pads the `circuits` in new
    ones with a number of qubits equal to the one provided by the platform.
    At the end, the circuits are transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns the list of transpiled circuits and the execution results.
    """
    transpiled_circuits = transpile_circuits(
        circuits,
        qubit_maps,
        backend,
        transpiler,
    )
    return transpiled_circuits, backend.execute_circuits(
        transpiled_circuits, initial_states=initial_states, nshots=nshots
    )


def execute_transpiled_circuit(
    circuit: Circuit,
    qubit_map: list[QubitId],
    backend: Backend,
    transpiler: Optional[Passes],
    initial_state=None,
    nshots=1000,
):
    """Transpile `circuit`.

    If the `qibolab` backend is used, this function pads the `circuit` in new a
    one with a number of qubits equal to the one provided by the platform.
    At the end, the circuit is transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns the transpiled circuit and the execution results.
    """

    transpiled_circ = transpile_circuits(
        [circuit],
        [qubit_map],
        backend,
        transpiler,
    )[0]
    return transpiled_circ, backend.execute_circuit(
        transpiled_circ, initial_state=initial_state, nshots=nshots
    )


def get_natives(platform):
    """
    Return the list of native gates defined in the `platform`.
    This function assumes the native gates to be the same for each
    qubit and pair.
    """
    pairs = list(platform.pairs.values())[0]
    qubit = list(platform.qubits.values())[0]
    two_qubit_natives = list(pairs.native_gates.raw.keys())
    single_qubit_natives = list(qubit.native_gates.raw.keys())
    # Solve Qibo-Qibolab mismatch
    single_qubit_natives.append("RZ")
    single_qubit_natives.append("Z")
    single_qubit_natives.remove("RX12")
    replacements = {
        "RX": "GPI2",
        "MZ": "M",
    }
    new_single_natives = [replacements.get(i, i) for i in single_qubit_natives]
    natives = new_single_natives + two_qubit_natives
    return natives


def set_compiler(backend, natives):
    """
    Set the compiler to execute the native gates defined by the platform.
    """
    compiler = backend.compiler
    for native in natives:
        gate = getattr(gates, native)
        if gate not in compiler.rules:

            def rule(qubits_ids, platform, parameters=None):
                if len(qubits_ids[1]) == 1:
                    native_gate = platform.qubits[tuple(qubits_ids[1])].native_gates
                else:
                    native_gate = platform.pairs[tuple(qubits_ids[1])].native_gates
                pulses = getattr(native_gate, native).pulses
                return PulseSequence(pulses), {}

            backend.compiler[gate] = rule


def dummy_transpiler(backend: Backend) -> Passes:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise None.
    """
    platform = backend.platform
    native_gates = get_natives(platform)
    set_compiler(backend, native_gates)
    native_gates = list(map(lambda x: getattr(gates, x), native_gates))
    unroller = Unroller(NativeGates.from_gatelist(native_gates))
    return Passes(connectivity=platform.topology, passes=[unroller])


def pad_circuit(nqubits, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where the i-th physical qubit is mapped
    into the 0th logical qubit and so on.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
