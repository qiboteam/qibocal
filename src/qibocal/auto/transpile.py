# This file contains functions to transpile and execute quantum circuits.
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import cycle

import numpy as np
from qibo import Circuit, gates
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qibolab import AcquisitionType, AveragingMode, Platform, PulseSequence
from qibolab._core.compilers import Compiler
from qibolab._core.identifier import Result
from qibolab._core.native import NativeContainer
from qibolab._core.pulses import PulseId

from qibocal.auto.operation import QubitId

REPLACEMENTS = {
    "RX": "GPI2",
    "MZ": "M",
}

QubitMap = list[QubitId]
"""An array where the elements are physical qubit IDs (str/int) and the indices are
logical qubit IDs
"""
ResultMap = dict[QubitId | tuple[QubitId, ...], list[Counter[str]]]
"""A dictionary mapping the physical qubit ID(s) measured to an array of state counts
per requested measurement
"""


def _string_to_integer_qubit_maps(
    qubit_maps: list[QubitMap], platform: Platform
) -> list[list[int]]:
    """QubitId can be integers or strings. If the qubit maps contain string IDs, we
    convert them to integer indices based on the platform's qubit order.
    """
    qubits = list(platform.qubits)
    return [
        [q if isinstance(q, int) else qubits.index(q) for q in qubit_map]
        for qubit_map in qubit_maps
    ]


def _validate_measurement(
    gate: gates.M, sequence: PulseSequence, readout: dict[PulseId, Result]
):
    """Validate measurement gate and sequence consistency."""
    for _, acquisition in sequence.acquisitions:
        if acquisition.id not in readout:
            raise KeyError(
                f"Acquisition ID {acquisition.id} not found in readout results."
            )
    assert len(gate.qubits) == len(sequence.acquisitions)


def _resolve_results_mapping_singleshot(
    platform_qubit_map: QubitMap,
    readout: dict[PulseId, Result],
    measurement_map: dict[gates.M, PulseSequence],
) -> ResultMap:
    """
    Iterates across the requested measurements and fetches the corresponding results as a count of states.
    If a multi-qubit measurement is requested, reconcile the results per shot into a multi-qubit state count.

    Args:
        platform_qubit_map: A list where the indices are logical qubit indices and the values are physical qubit IDs.
        readout: Results from circuit execution.
        measurement_map: Map of measurement registers to measurement pulse sequences for the current circuit.

    Returns:
        A dictionary mapping the physical qubit IDs measured to an array of state counters for the current circuit.
    """

    measurements: ResultMap = defaultdict(list)
    for measure, sequence in measurement_map.items():
        _validate_measurement(measure, sequence, readout)
        if len(measure.qubits) == 1:
            qid = platform_qubit_map[measure.qubits[0]]
        else:
            qid = tuple([platform_qubit_map[qubit_id] for qubit_id in measure.qubits])
        arr = np.stack(
            [readout[pulse.id] for (_, pulse) in sequence.acquisitions]
        ).astype(int)
        measurements[qid].append(Counter("".join(map(str, col)) for col in arr.T))

    return measurements


def _resolve_results_mapping_averaged(
    platform_qubit_map: QubitMap,
    readout: dict[PulseId, Result],
    measurement_map: dict[gates.M, PulseSequence],
    nshots: int,
) -> ResultMap:
    """
    Iterates across the requested measurements and fetches the corresponding results as a count of states.

    Args:
        platform_qubit_map: A list where the indices are logical qubit indices and the values are physical qubit IDs.
        readout: Results from circuit execution.
        measurement_map: Map of measurement registers to measurement pulse sequences for the current circuit.
        nshots: Number of shots requested.

    Returns:
        A dictionary mapping the physical qubit IDs measured to an array of state counters for the current circuit.
    """

    measurements: ResultMap = defaultdict(list)
    for measure, sequence in measurement_map.items():
        _validate_measurement(measure, sequence, readout)
        if len(measure.qubits) != 1:
            raise ValueError(
                "Hardware averaging is only supported for single qubit readout."
            )
        qid = platform_qubit_map[measure.qubits[0]]
        excited_frac = readout[sequence.acquisitions[0][1].id]
        measurements[qid].append(
            Counter(
                {
                    "0": int(np.round((1 - excited_frac) * nshots)),
                    "1": int(np.round(excited_frac * nshots)),
                }
            )
        )

    return measurements


def _execute_circuits(
    platform: Platform,
    compiler: Compiler,
    circuits: list[Circuit],
    nshots: int,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[ResultMap]:
    """Executes multiple quantum circuits with a single communication with
    the control electronics.

    Circuits are unrolled to a single pulse sequence.
    """

    sequences, measurement_maps = zip(
        *(compiler.compile(circuit, platform) for circuit in circuits)
    )

    # acquisition_type is always DISCRIMINATION for circuits.
    readout = platform.execute(
        list(sequences),
        nshots=nshots,
        averaging_mode=averaging_mode,
        acquisition_type=AcquisitionType.DISCRIMINATION,
    )

    platform_qubit_mapping = list(platform.qubits)
    if averaging_mode.average:
        measurements_per_circuit = [
            _resolve_results_mapping_averaged(
                platform_qubit_map=platform_qubit_mapping,
                readout=readout,
                measurement_map=measurement_map,
                nshots=nshots,
            )
            for measurement_map in measurement_maps
        ]
    else:
        measurements_per_circuit = [
            _resolve_results_mapping_singleshot(
                platform_qubit_map=platform_qubit_mapping,
                readout=readout,
                measurement_map=measurement_map,
            )
            for measurement_map in measurement_maps
        ]

    assert len(measurements_per_circuit) == len(circuits)
    return measurements_per_circuit


def execute_circuits(
    circuits: list[Circuit],
    qubit_maps: list[QubitMap],
    platform: Platform,
    transpiler: Passes,
    compiler: Compiler,
    nshots: int,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[ResultMap]:
    """Execute multiple quantum circuits.

    The circuits are first padded to fit the number of qubits in the platform
    and rearranged according to the qubit mapping before being transpiled and executed.

    Args:
        circuits: List of quantum circuits to transpile and execute.
        platform: The platform to transpile circuits for and execute on.
        transpiler: The transpiler to apply to the circuits.
        compiler: The compiler to use for circuit compilation.
        nshots: Number of times to sample from the experiment.
        qubit_map: A mapping of physical qubit IDs to logical qubit indices.
        qubit_maps: An array of physical qubit to logical qubit mapping per circuit.
        averaging_mode: Averaging mode for measurements. Default is single-shot.

    Returns:
        List of dictionaries mapping physical qubit ID(s) to measurement outcomes as Counter objects,
        one per circuit. Each Counter maps measurement outcome states as strings (e.g., "01", "10")
        to their occurrence counts. Total counts per counter equals nshots.

    Examples:
        .. testcode::

        from qibo import Circuit, gates
        from qibolab import create_platform
        from qibocal.auto.transpile import (
            build_native_gate_compiler,
            build_native_gate_transpiler,
            execute_circuits,
        )

        platform = create_platform("dummy")
        transpiler = build_native_gate_transpiler(platform)
        compiler = build_native_gate_compiler(platform)

        circuit = Circuit(1)
        circuit.add(gates.M(0))

        qubit = next(iter(platform.qubits))
        [results] = execute_circuits(
            circuits=[circuit],
            qubit_maps=[[qubit]],
            platform=platform,
            transpiler=transpiler,
            compiler=compiler,
            nshots=100,
        )
        [counts] = results[qubit]

        assert sum(counts.values()) == 100
    """

    assert len(qubit_maps) == 1 or len(qubit_maps) == len(circuits)

    transpiled = [Circuit(platform.nqubits) for _ in circuits]
    _qubit_maps = _string_to_integer_qubit_maps(qubit_maps, platform)

    for actual_circuit, original_circuit, qubit_map in zip(
        transpiled, circuits, cycle(_qubit_maps)
    ):
        transpiled_circ, _ = transpiler(original_circuit)
        actual_circuit.add(transpiled_circ.on_qubits(*qubit_map))

    return _execute_circuits(
        platform,
        compiler,
        transpiled,
        nshots=nshots,
        averaging_mode=averaging_mode,
    )


def _natives(platform: Platform) -> dict[str, NativeContainer]:
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


def _create_rule(name: str, natives: NativeContainer) -> Callable:
    """Create rule for gate name given container natives."""

    def rule(gate: gates.Gate, natives: NativeContainer) -> PulseSequence:
        return natives.ensure(name).create_sequence()

    return rule


def build_native_gate_compiler(platform: Platform) -> Compiler:
    """Build a compiler that follows the native gates defined by `platform`.

    Starting from :meth:`Compiler.default`, this function overrides and extends
    gate rules using the native containers available on the platform so circuit
    compilation is consistent with the selected hardware configuration.

    Args:
        platform: The quantum platform containing native gate definitions.

    Returns:
        A Compiler instance with rules set according to the platform's native gates.
    """
    native_gates = _natives(platform)
    compiler = Compiler.default()
    rules = {}
    for name, natives_container in native_gates.items():
        gate = getattr(gates, name)
        if gate not in compiler.rules:
            rules[gate] = _create_rule(name, natives_container)
        else:
            rules[gate] = compiler.rules[gate]
    rules[gates.I] = compiler.rules[gates.I]
    return Compiler(rules=rules)


def build_native_gate_transpiler(platform: Platform) -> Passes:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise `None`. This function overwrites the compiler defined in the
    backend, taking into account the native gates defined in the`platform` (see
    :func:`build_native_gate_compiler`).

    Args:
        platform: The quantum platform containing native gate definitions.

    Returns:
        A Passes instance with an unroller set according to the platform's native gates.
    """
    native_gates = _natives(platform)
    native_gates = [getattr(gates, x) for x in native_gates]
    unroller = Unroller(NativeGates.from_gatelist(native_gates))
    return Passes(connectivity=platform.pairs, passes=[unroller])
