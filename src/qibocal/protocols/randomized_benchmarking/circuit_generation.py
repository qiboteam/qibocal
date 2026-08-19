"""Circuit generation helpers for randomized benchmarking."""

import pathlib

import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibolab import AveragingMode

from qibocal.auto.operation import Parameters, QubitId, QubitPairId
from qibocal.auto.transpile import (
    build_native_gate_compiler,
    build_native_gate_transpiler,
    execute_circuits,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.dict_utils import (
    SINGLE_QUBIT_CLIFFORDS_NAMES,
    calculate_pulses_clifford,
    clifford2gates,
    find_cliffords,
    generate_inv_dict_cliffords_file,
    load_cliffords,
    separator,
)

from .types import (
    GLOBAL_PHASES,
    NPULSES_PER_CLIFFORD,
    SINGLE_QUBIT_CLIFFORDS,
    CircuitIndex,
    IndexedCircuit,
    IndexedResult,
    RB2QData,
    RB2QInterData,
    RBData,
)


def random_clifford(random_index_gen):
    """Generates random Clifford operator."""

    random_index = int(random_index_gen(SINGLE_QUBIT_CLIFFORDS))
    clifford_gate = SINGLE_QUBIT_CLIFFORDS[random_index](0)

    return clifford_gate


def random_2q_clifford(random_index_gen, two_qubit_cliffords):
    """Generates random two qubit Clifford operator."""

    random_index = int(random_index_gen(two_qubit_cliffords))
    clifford = two_qubit_cliffords[str(random_index)]
    clifford_gate = clifford2gates(clifford)

    return clifford_gate


class RBGenerator:
    """
    This class generates random two qubit cliffords for randomized benchmarking.
    """

    def __init__(self, seed, file=None):
        self.seed = seed
        self.local_state = (
            np.random.default_rng(seed)
            if seed is None or isinstance(seed, int)
            else seed
        )

        if file is not None:
            self.two_qubit_cliffords = load_cliffords(file)
        self.file = file

    def random_index(self, gate_dict):
        """Generates a random index within the range of the given file len."""
        return self.local_state.integers(0, len(gate_dict))

    def random_layer_gen_single_qubit(self):
        """Generates a random single-qubit clifford gate."""
        return random_clifford(self.random_index)

    def random_layer_gen_two_qubit(self):
        """Generates a random two-qubit clifford gate."""
        return random_2q_clifford(self.random_index, self.two_qubit_cliffords)

    def calculate_average_pulses(self):
        """Average number of pulses per clifford."""
        # FIXME: Make it work for single qubit properly if we need it ?
        return (
            calculate_pulses_clifford(self.two_qubit_cliffords)
            if self.file is not None
            else NPULSES_PER_CLIFFORD
        )


def setup_data(
    params: Parameters,
    npulses_per_clifford: float,
    single_qubit: bool = True,
    interleave: str | None = None,
):
    """
    Set up the randomized benchmarking experiment data class.

    Args:
        params (Parameters): The parameters for the experiment.
        single_qubit (bool, optional): Flag indicating whether the experiment is for a single qubit or two qubits. Defaults to True.
        interleave: (str, optional): The type of interleaving to apply. Defaults to None.

    Returns:
        data: The experiment data class.
    """
    # Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    if single_qubit:
        cls = RBData
    elif interleave is not None:
        cls = RB2QInterData
    else:
        cls = RB2QData
    data = cls(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
        npulses_per_clifford=npulses_per_clifford,
    )
    return data


def _generate_indexed_circuits(
    params: Parameters,
    rb_gen: RBGenerator,
    targets: list[QubitId] | list[QubitPairId],
    inverse_layer: bool = True,
    interleave: str | None = None,
) -> list[IndexedCircuit]:
    """Generate randomized benchmarking circuits with explicit indexing of
    (qubit, depth, iteration) coordinates.

    Args:
        params: Experiment parameters containing depths, niter.
        rb_gen: RBGenerator instance to use for generating Clifford gates.
        targets: List of target qubit IDs.
        inverse_layer: Whether to add an inverse layer to the circuits. Defaults to True.
        interleave: Interleaving pattern for the circuits. Defaults to None.

    Returns:
        List of IndexedCircuit objects with explicit (qubit, depth, iteration) metadata.
    """
    indexed_circuits: list[IndexedCircuit] = []

    inv_file = getattr(params, "file_inv", None)

    assert len(targets) > 0
    two_qubit = isinstance(targets[0], tuple)
    nqubits = len(targets) * (2 if two_qubit else 1)

    target_id_map = {
        target: (idx * 2, idx * 2 + 1) if two_qubit else (idx,)
        # Reverse assignment for little-endianess resolution
        for idx, target in enumerate(reversed(targets))
    }

    for iteration in range(params.niter):
        for depth in params.depths:
            full_circuit = Circuit(nqubits)
            for target in targets:
                circuit = layer_circuit(rb_gen, depth, target, interleave)
                if inverse_layer:
                    add_inverse_layer(circuit, rb_gen, inv_file)
                full_circuit.add(circuit.on_qubits(*target_id_map[target]))

            add_measurement_layer(full_circuit)
            index = CircuitIndex(
                depth=depth,
                iteration=iteration,
            )
            indexed_circuits.append(IndexedCircuit(circuit=full_circuit, index=index))

    return indexed_circuits


def _execute_indexed_circuits(
    indexed_circuits: list[IndexedCircuit],
    params: Parameters,
    platform: CalibrationPlatform,
    qubit_map: list[QubitId],
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[IndexedResult]:
    """Execute indexed circuits and return results paired with their indices.

    Args:
        indexed_circuits: List of IndexedCircuit objects to execute.
        params: Experiment parameters.
        platform: CalibrationPlatform to execute on.
        qubit_map: List of physical qubit IDs.

    Returns:
        List of IndexedResult objects with execution results paired with their indices.
    """

    qubit_maps = [qubit_map] * len(indexed_circuits)
    circuits = []
    for indexed_circuit in indexed_circuits:
        circuits.append(indexed_circuit.circuit)

    transpiler = build_native_gate_transpiler(platform)
    compiler = build_native_gate_compiler(platform)

    executed_results = execute_circuits(
        circuits,
        qubit_maps,
        platform,
        transpiler,
        compiler,
        nshots=params.nshots,
        averaging_mode=averaging_mode,
    )

    indexed_results = [
        IndexedResult(result=result, index=ic.index)
        for ic, result in zip(indexed_circuits, executed_results)
    ]

    return indexed_results


def layer_circuit(
    rb_gen: RBGenerator,
    depth: int,
    target: QubitId | QubitPairId,
    interleave: str | None = None,
) -> Circuit:
    """Creates a circuit of `depth` layers from a generator `layer_gen` yielding `Circuit` or `Gate`
    and a dictionary with random indexes used to select the clifford gates.

    Args:
        layer_gen (Callable): Should return gates or a full circuit specifying a layer.
        depth (int): Number of layers.
        interleave (str, optional): Interleaving pattern for the circuits. Defaults to None.

    Returns:
        Circuit: with `depth` many layers.
    """
    full_circuit = None
    if isinstance(target, (str, int)):
        nqubits = 1
        rb_gen_layer = rb_gen.random_layer_gen_single_qubit
    elif isinstance(target, tuple):  # Tuple for qubit pair
        nqubits = 2
        rb_gen_layer = rb_gen.random_layer_gen_two_qubit
    else:
        raise NotImplementedError("RB with more than 2 qubits is not implemented")
    # Build each layer, there will be depth many in the final circuit.

    for _ in range(depth):
        # Generate a layer.
        new_layer = rb_gen_layer()
        new_circuit = Circuit(nqubits)
        if nqubits == 1:
            new_circuit.add(new_layer)
        elif nqubits == 2:
            for gate in new_layer:
                new_circuit.add(gate)
            # FIXME: General interleave
            if interleave == "CZ":
                interleaved_clifford = rb_gen.two_qubit_cliffords["13"]
                interleaved_clifford_gate = clifford2gates(interleaved_clifford)
                new_circuit.add(interleaved_clifford_gate)

        if full_circuit is None:  # instantiate in first loop
            full_circuit = Circuit(new_circuit.nqubits)
        full_circuit += new_circuit
    return full_circuit


def add_inverse_layer(
    circuit: Circuit, rb_gen: RBGenerator, file_inv: pathlib.Path | None = None
):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """
    if file_inv:  # if file_inv is not none, it is for a two qubit gate circuit
        two_qubit_cliffords = rb_gen.two_qubit_cliffords
        path = pathlib.Path(__file__).parent / file_inv
        if file_inv is None and not path.is_file():
            clifford_matrices_inv = generate_inv_dict_cliffords_file(
                two_qubit_cliffords, file_inv
            )
        else:
            clifford_matrices_inv = np.load(path)

        if circuit.depth > 0:
            clifford = circuit.unitary()

            cliffords = [clifford * global_phase for global_phase in GLOBAL_PHASES]
            cliffords_inv = [np.linalg.inv(clifford).round(3) for clifford in cliffords]
            index_inv = None
            for clifford_inv in cliffords_inv:
                clifford_inv += 0.0 + 0.0j
                clifford_inv_str = np.array2string(clifford_inv, separator=",")
                if clifford_inv_str in clifford_matrices_inv.files:
                    index_inv = clifford_matrices_inv[clifford_inv_str]

            clifford = two_qubit_cliffords[str(index_inv)]

            gate_list = clifford.split(",")

            clifford_list = find_cliffords(gate_list)

            clifford_gate = []
            for clifford in clifford_list:
                values_with_1, values_with_2, value_with_CZ = separator(clifford)
                clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_1](0))
                clifford_gate.append(SINGLE_QUBIT_CLIFFORDS_NAMES[values_with_2](1))
                if value_with_CZ:
                    clifford_gate.append(gates.CZ(0, 1))

            for gate in clifford_gate:
                circuit.add(gate)
    else:  # single qubit gate circuit
        if circuit.depth > 0:
            circuit.add(
                gates.Unitary(circuit.unitary(), *range(circuit.nqubits)).dagger()
            )


def add_measurement_layer(circuit: Circuit):
    """Adds a measurement layer at the end of the circuit.

    Args:
        circuit (Circuit): Measurement gates added in place to end of this circuit.
    """

    for qubit in range(circuit.nqubits):
        circuit.add(gates.M(qubit))
