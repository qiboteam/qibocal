from typing import Optional

from qibo import Circuit
from qibo.backends import construct_backend
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qibolab.platform import Platform
from qibolab.qubits import QubitId


def transpile_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    platform: Platform,
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
    platform: Platform,
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
        platform,
        transpiler,
    )
    backend = construct_backend(backend="qibolab", platform=platform)
    return transpiled_circuits, backend.execute_circuits(
        transpiled_circuits, initial_states=initial_states, nshots=nshots
    )


def execute_transpiled_circuit(
    circuit: Circuit,
    qubit_map: list[QubitId],
    platform: Platform,
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
        platform,
        transpiler,
    )[0]

    backend = construct_backend(backend="qibolab", platform=platform)
    return transpiled_circ, backend.execute_circuit(
        transpiled_circ, initial_state=initial_state, nshots=nshots
    )


def dummy_transpiler(platform) -> Optional[Passes]:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise None.
    """
    unroller = Unroller(NativeGates.default())
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
