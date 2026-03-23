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
    qubit_maps: list[list[QubitId]],
    nshots: int = 1000,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[Counter[str]]:
    """Executes multiple quantum circuits with a single communication with
    the control electronics.

    Circuits are unrolled to a single pulse sequence.

    Args:
        platform: The quantum platform to execute circuits on.
        compiler: The compiler to use for circuit compilation.
        circuits: List of quantum circuits to execute.
        qubit_maps: List of qubit maps, one per circuit. Each qubit map maps physical
            qubit IDs to logical qubit indices.
        nshots: Number of times to sample from the experiment. Default is 1000.
        averaging_mode: Averaging mode for measurements. Only 'SINGLESHOT' supports
            multiple qubits and multi-qubit readout. 'CYCLIC' is single-qubit only and
            uses hardware average. Default is SINGLESHOT.
        Default is SINGLESHOT (CYCLIC only works with a single qubit).

    Returns:
        List of measurement outcome as Counter objects, one per circuit. Each Counter
        maps measurement outcome states as strings (e.g., "01", "10") to their
        occurrence counts. Total counts per counter equals nshots.

    Examples:
        .. testcode::

        from qibo import Circuit, gates
        from qibolab import create_platform
        from qibocal.auto.transpile import execute_circuits, set_compiler

        platform = create_platform("dummy")
        compiler = set_compiler(platform)

        circuit = Circuit(1)
        circuit.add(gates.M(0))

        qubit = next(iter(platform.qubits))
        [counts] = execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=[circuit],
            qubit_maps=[[qubit]],
            nshots=100,
        )

        assert sum(counts.values()) == 100
    """

    # TODO: Maybe these loops can be parallelized
    sequences, measurement_maps = zip(
        *(compiler.compile(circuit, platform) for circuit in circuits)
    )

    # TODO?: pass options dict
    readout = platform.execute(sequences, nshots=nshots, averaging_mode=averaging_mode)

    if averaging_mode.average and len(qubit_maps[0]) > 1:
        raise ValueError(
            "Averaging mode CYCLIC only supports single qubit readout. "
            "Use SINGLESHOT instead. The reason is that the excited state probability "
            "individual qubits (which is what CYCLIC extracts) does not provide full "
            "information about the probability distribution of the full set of basis "
            "states in a multi-qubit setup."
        )

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
                        "0": int(np.round((1 - excited_frac) * nshots)),
                        "1": int(np.round(excited_frac * nshots)),
                    }
                )
            )
    else:
        for qubit_map, measurement_map in zip(qubit_maps, measurement_maps):
            assert len(qubit_map) == len(measurement_map)
            # The mapping from physical to logical qubits
            phys_to_logic_mapping = {q: i for i, q in enumerate(qubit_map)}
            result = {}
            for gate, sequence in measurement_map.items():
                # assert that a single measurement gate only measures the state of a single qubit
                assert len(gate.qubits) == 1
                assert len(sequence.acquisitions) == 1
                logical_qubit = phys_to_logic_mapping[gate.qubits[0]]
                result[logical_qubit] = readout[sequence.acquisitions[0][1].id]
            # The inverse sorting is to have little-endian bitstring notation, which
            # means that the qubit with the smalles qubitId is the most significant bit
            # in the output string (on the right).
            invsorted_result = sorted(result)[::-1]
            arr = np.stack([result[q] for q in invsorted_result]).astype(int)
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

    Args:
        platform: The quantum platform containing native gate definitions.

    Returns:
        A Compiler instance with rules set according to the platform's native gates.
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

    Args:
        platform: The quantum platform containing native gate definitions.

    Returns:
        A Passes instance with an unroller set according to the platform's native gates.
    """
    native_gates = natives(platform)
    native_gates = [getattr(gates, x) for x in native_gates]
    unroller = Unroller(NativeGates.from_gatelist(native_gates))
    return Passes(connectivity=platform.pairs, passes=[unroller])


def pad_circuit(nqubits: int, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where physical qubit i is mapped into the
    0th logical qubit and so on.

    Args:
        nqubits: The total number of qubits in the new circuit.
        circuit: The original quantum circuit to be padded.
        qubit_map: A list mapping physical qubits to logical qubits in the new circuit.

    Returns:
        A Circuit instance with `nqubits` qubits, containing the original circuit's
        gates mapped according to `qubit_map`.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
