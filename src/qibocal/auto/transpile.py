# This file contains functions to transpile and execute quantum circuits.
#
# TODO: Since these functions are always used in the same way, we should probably
# provide a single function that takes care of setting the compiler, transpiler, doing
# the transpilation and execution in a single call. This would mean that set_compiler
# and dummy_transpiler are called for every circuit exeuction instead of just once per
# protocol, so I'm not convinced that's what should be done.
from collections import Counter
from typing import Callable

import numpy as np
from qibo import Circuit, gates
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qibolab import AveragingMode, Platform, PulseSequence
from qibolab._core.compilers import Compiler
from qibolab._core.native import NativeContainer

from qibocal.auto.operation import QubitId

REPLACEMENTS = {
    "RX": "GPI2",
    "MZ": "M",
}


def transpile_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    platform: Platform,
    transpiler: Passes,
) -> list[Circuit]:
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
    # QubitId can be integers or strings. pad_circuit only works with integer qubit IDs,
    # so if the qubit maps contain string IDs, we convert them to integer indices based
    # on the platform's qubit order.
    _qubit_maps: list[list[int]] = [
        [q if isinstance(q, int) else qubits.index(q) for q in qubit_map]
        for qubit_map in qubit_maps
    ]
    platform_nqubits = platform.nqubits
    for circuit, qubit_map in zip(circuits, _qubit_maps):
        new_circuit = pad_circuit(platform_nqubits, circuit, qubit_map)
        transpiled_circ, _ = transpiler(new_circuit)
        transpiled_circuits.append(transpiled_circ)

    return transpiled_circuits


def execute_circuits(
    platform: Platform,
    compiler: Compiler,
    circuits: list[Circuit],
    nshots: int = 1000,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[Counter[str]]:
    """Executes multiple quantum circuits with a single communication with
    the control electronics.

    Circuits are unrolled to a single pulse sequence.

    Args:
        circuits (list): List of circuits to execute.
        nshots (int): Number of shots to sample from the experiment.

    Returns:
        List of ``MeasurementOutcomes`` objects containing the results acquired from the execution of each circuit.
    """

    # TODO: Maybe these loops can be parallelized
    sequences, measurement_maps = zip(
        *(compiler.compile(circuit, platform) for circuit in circuits)
    )

    # TODO?: pass options dict
    readout = platform.execute(sequences, nshots=nshots, averaging_mode=averaging_mode)

    countslist = []
    if averaging_mode.average:
        # NOTE: averaging mode only makes sense for a two state readout. If ther eare
        # more states it would have to be conditional since the excited state probablity
        # of idividual qubits does not provide full information about the probablity
        # distribution of the full set of basis states.
        for excited_frac in readout.values():
            countslist.append(
                Counter(
                    {
                        "0": np.round((1 - excited_frac) * nshots),
                        "1": np.round(excited_frac * nshots),
                    }
                )
            )
    else:
        for measurement_map in measurement_maps:
            result = {}
            for gate, sequence in measurement_map.items():
                # assert that a single measurement gate only measures the state of a single qubit
                assert len(gate.qubits) == 1
                assert len(sequence.acquisitions) == 1
                result[gate.qubits[0]] = readout[sequence.acquisitions[0][1].id]
            arr = np.stack([result[q] for q in sorted(result)]).astype(int)
            countslist.append(Counter("".join(map(str, col)) for col in arr.T))

    assert all(sum(counts.values()) == nshots for counts in countslist)

    return countslist


def natives(platform: Platform) -> dict[str, NativeContainer]:
    """
    Return the dict of native gates name with the associated native container
    defined in the `platform`. This function assumes the native gates to be the same for each
    qubit and pair.
    """
    qubit = next(iter(platform.qubits))
    single_qubit_natives_container = platform.natives.single_qubit[qubit]
    single_qubit_natives = list(single_qubit_natives_container.model_fields)
    if len(platform.pairs) > 0:
        # add two qubit natives only if there are pairs
        pair = next(iter(platform.pairs))
        two_qubit_natives_container = platform.natives.two_qubit[pair]
        two_qubit_natives = list(two_qubit_natives_container.model_fields)
    else:
        two_qubit_natives = []
    # Solve Qibo-Qibolab mismatch
    single_qubit_natives.append("RZ")
    single_qubit_natives.append("Z")
    single_qubit_natives.remove("RX12")
    single_qubit_natives.remove("RX90")
    single_qubit_natives.remove("CP")
    single_qubit_natives = [REPLACEMENTS.get(x, x) for x in single_qubit_natives]
    return {i: platform.natives.single_qubit[qubit] for i in single_qubit_natives} | {
        i: platform.natives.two_qubit[pair] for i in two_qubit_natives
    }


def create_rule(name: str, natives: NativeContainer) -> Callable:
    """Create rule for gate name given container natives."""

    def rule(gate: gates.Gate, natives: NativeContainer) -> PulseSequence:
        return natives.ensure(name).create_sequence()

    return rule


def set_compiler(platform: Platform) -> Compiler:
    """Build a compiler that follows the native gates defined by `platform`.

    Starting from :meth:`Compiler.default`, this function overrides and extends
    gate rules using the native containers available on the platform so circuit
    compilation is consistent with the selected hardware configuration.
    """
    native_gates = natives(platform)
    compiler = Compiler.default()
    rules = {}
    for name, natives_container in native_gates.items():
        gate = getattr(gates, name)
        if gate not in compiler.rules:
            rules[gate] = create_rule(name, natives_container)
        else:
            rules[gate] = compiler.rules[gate]
    rules[gates.I] = compiler.rules[gates.I]
    return Compiler(rules=rules)


def dummy_transpiler(platform: Platform) -> Passes:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise `None`. This function overwrites the compiler defined in the
    backend, taking into account the native gates defined in the`platform` (see
    :func:`set_compiler`).
    """
    native_gates = natives(platform)
    native_gates = [getattr(gates, x) for x in native_gates]
    unroller = Unroller(NativeGates.from_gatelist(native_gates))
    return Passes(connectivity=platform.pairs, passes=[unroller])


def pad_circuit(nqubits: int, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where the i-th physical qubit is mapped
    into the 0th logical qubit and so on.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
