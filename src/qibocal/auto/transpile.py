from typing import Optional

from qibo import Circuit
from qibo.backends.abstract import Backend
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller


def execute_transpiled_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[int]],
    backend: Backend,
    initial_states=None,
    nshots=1000,
    transpiler: Optional[Passes] = None,
):
    """
    If the `qibolab` backend is used, this function pads the `circuits` in new
    ones with a number of qubits equal to the one provided by the platform.
    At the end, the circuits are transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns the list of transpiled circuits and the execution results.
    """
    new_circuits = []
    if backend.name == "qibolab":
        platform_nqubits = backend.platform.nqubits
        for circuit, qubit_map in zip(circuits, qubit_maps):
            new_circuit = pad_circuit(platform_nqubits, circuit, qubit_map)
            transpiled_circ, _ = transpiler(new_circuit)
            new_circuits.append(transpiled_circ)
    else:
        new_circuits = circuits
    return new_circuits, backend.execute_circuits(
        new_circuits, initial_states=initial_states, nshots=nshots
    )


def execute_transpiled_circuit(
    circuit: Circuit,
    qubit_map: list[int],
    backend: Backend,
    initial_state=None,
    nshots=1000,
    transpiler: Optional[Passes] = None,
):
    """
    If the `qibolab` backend is used, this function pads the `circuit` in new a
    one with a number of qubits equal to the one provided by the platform.
    At the end, the circuit is transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns the transpiled circuit and the execution results.
    """
    if backend.name == "qibolab":
        platform_nqubits = backend.platform.nqubits
        new_circuit = pad_circuit(platform_nqubits, circuit, qubit_map)
        transpiled_circ, _ = transpiler(new_circuit)
    else:
        transpiled_circ = circuit
    return transpiled_circ, backend.execute_circuit(
        transpiled_circ, initial_state=initial_state, nshots=nshots
    )


def dummy_transpiler(backend) -> Optional[Passes]:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise None.
    """
    if backend.name == "qibolab":
        unroller = Unroller(NativeGates.default())
        return Passes(connectivity=backend.platform.topology, passes=[unroller])
    return None


def pad_circuit(nqubits, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where the i-th physical qubit is mapped
    into the 0th logical qubit and so on.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
