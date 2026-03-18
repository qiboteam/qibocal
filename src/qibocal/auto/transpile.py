from collections import Counter
from typing import Optional, Sequence

import numpy as np
from qibo import Circuit, gates
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qibolab import AveragingMode, Platform, PulseSequence
from qibolab._core.compilers import Compiler
from qibolab._core.native import NativeContainer

from qibocal.auto.operation import QubitId
from qibocal.config import raise_error

REPLACEMENTS = {
    "RX": "GPI2",
    "MZ": "M",
}


def transpile_circuits(
    circuits: list[Circuit],
    qubit_maps: list[Sequence[QubitId]],
    platform,
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


def execute_circuits(
    platform: Platform,
    compiler: Compiler,
    circuits: list[Circuit],
    initial_states: Optional[Circuit] = None,
    nshots: int = 1000,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[Counter[str]]:
    """Executes multiple quantum circuits with a single communication with
    the control electronics.

    Circuits are unrolled to a single pulse sequence.

    Args:
        circuits (list): List of circuits to execute.
        initial_states (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
            If ``None`` the default ``|00...0>`` state is used.
        nshots (int): Number of shots to sample from the experiment.

    Returns:
        List of ``MeasurementOutcomes`` objects containing the results acquired from the execution of each circuit.
    """
    if isinstance(initial_states, Circuit):
        return execute_circuits(
            platform=platform,
            compiler=compiler,
            circuits=[initial_states + circuit for circuit in circuits],
            nshots=nshots,
            averaging_mode=averaging_mode,
        )
    if initial_states is not None:
        raise_error(
            ValueError,
            "Hardware backend only supports circuits as initial states.",
        )

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

    assert all(sum(counts.values()) == nshots for counts in countslist), (
        "The sum of shots in all possible outcomes should be equal to nshots."
    )

    return countslist


# TODO: I don't like this. The name suggests it executes transpiled circuits, but it
# also transpiles them. There is no benefit turning two actions into a single function.
def execute_transpiled_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    platform,
    compiler,
    transpiler: Optional[Passes],
    initial_states=None,
    nshots=1000,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
):
    """Transpile `circuits`.

    If the `qibolab` backend is used, this function pads the `circuits` in new
    ones with a number of qubits equal to the one provided by the platform.
    At the end, the circuits are transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns a list of the execution results.
    """
    transpiled_circuits = transpile_circuits(
        circuits,
        qubit_maps,
        platform,
        transpiler,
    )
    return execute_circuits(
        platform,
        compiler,
        transpiled_circuits,
        initial_states=initial_states,
        nshots=nshots,
        averaging_mode=averaging_mode,
    )


# TODO: I don't like this. The name suggests it executes transpiled circuits, but it
# also transpiles them. There is no benefit turning two actions into a single function.
def execute_transpiled_circuit(
    circuit: Circuit,
    qubit_map: Sequence[QubitId],
    platform: Platform,
    compiler,
    transpiler: Optional[Passes],
    initial_state=None,
    nshots=1000,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
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

    transpiled_circs = transpile_circuits(
        [circuit],
        [qubit_map],
        platform,
        transpiler,
    )
    return transpiled_circs[0], execute_circuits(
        platform,
        compiler,
        transpiled_circs,
        initial_states=initial_state,
        nshots=nshots,
        averaging_mode=averaging_mode,
    )


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


def create_rule(name, natives):
    """Create rule for gate name given container natives."""

    def rule(gate: gates.Gate, natives: NativeContainer) -> PulseSequence:
        return natives.ensure(name).create_sequence()

    return rule


def get_compiler(platform):
    """
    Set the compiler to execute the native gates defined by the platform.
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


def dummy_transpiler(platform) -> Passes:
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


def pad_circuit(nqubits, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where the i-th physical qubit is mapped
    into the 0th logical qubit and so on.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
