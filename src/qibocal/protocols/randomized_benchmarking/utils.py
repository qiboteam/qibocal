import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from qibo import gates
from qibo.models import Circuit
from qibolab import AveragingMode

from qibocal.auto.operation import Data, Parameters, QubitId, QubitPairId, Results
from qibocal.auto.transpile import (
    dummy_transpiler,
    execute_circuits,
    set_compiler,
    transpile_circuits,
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
from qibocal.protocols.utils import significant_digit

from .fitting import fit_exp1B_func


@dataclass(frozen=True)
class CircuitIndex:
    """Tracks the (qubit, depth, iteration) CircuitIndex of a circuit."""

    qubit: Union[QubitId, QubitPairId]
    depth: int
    iteration: int


@dataclass
class IndexedCircuit:
    """A circuit paired with its (qubit, depth, iteration) CircuitIndex."""

    circuit: Circuit
    index: CircuitIndex


@dataclass
class IndexedResult:
    """An execution result paired with its (qubit, depth, iteration) CircuitIndex."""

    result: Counter
    index: CircuitIndex


SINGLE_QUBIT_CLIFFORDS = {
    # Virtual gates
    0: gates.I,
    1: lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z,
    2: lambda q: gates.U3(q, 0, np.pi / 2, 0),  # gates.RZ(q, np.pi / 2),
    3: lambda q: gates.U3(q, 0, -np.pi / 2, 0),  # gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.U3(q, np.pi, 0, np.pi),  # X,
    5: lambda q: gates.U3(q, np.pi, 0, 0),  # Y,
    # pi/2 rotations
    6: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),
    7: lambda q: gates.U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),
    8: lambda q: gates.U3(q, np.pi / 2, 0, 0),
    9: lambda q: gates.U3(q, -np.pi / 2, 0, 0),
    # 2pi/3 rotations
    10: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    11: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
    12: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
    13: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
    14: lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    15: lambda q: gates.U3(q, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
    16: lambda q: gates.U3(q, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
    17: lambda q: gates.U3(q, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    18: lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    19: lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    20: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
    21: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
    22: lambda q: gates.U3(q, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    23: lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}

NPULSES_PER_CLIFFORD = 1.875

"""
Global phases that could appear in the Clifford group we defined in the "2q_cliffords.json" file
due to the gates we selected to generate the Clifford group.
"""
GLOBAL_PHASES = [
    1 + 0j,
    -1 + 0j,
    0 + 1j,
    0 - 1j,
    0.707 + 0.707j,
    -0.707 + 0.707j,
    0.707 - 0.707j,
    -0.707 - 0.707j,
]


RBType = np.dtype(
    [
        ("survival_probs", np.float64),
    ]
)
"""Custom dtype for RB."""


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


def number_to_str(
    value: Number,
    uncertainty: Optional[Union[Number, list, tuple, np.ndarray]] = None,
    precision: Optional[int] = None,
):
    """Converts a number into a string.

    Args:
        value (Number): the number to display
        uncertainty (Number or list or tuple or np.ndarray, optional): number or 2-element
            interval with the low and high uncertainties of ``value``. Defaults to ``None``.
        precision (int, optional): nonnegative number of floating points of the displayed value.
            If ``None``, defaults to the second significant digit of ``uncertainty``
            or ``3`` if ``uncertainty`` is ``None``. Defaults to ``None``.

    Returns:
        str: The number expressed as a string, with the uncertainty if given.
    """

    # If uncertainty is not given, return the value with precision
    if uncertainty is None:
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"

    if isinstance(uncertainty, Number):
        if precision is None:
            precision = (significant_digit(uncertainty) + 1) or 3
        return f"{value:.{precision}f} \u00b1 {uncertainty:.{precision}f}"

    # If any uncertainty is None, return the value with precision
    if any(u is None for u in uncertainty):
        return f"{value:.{precision if precision is not None else 3}f}"

    # If precision is None, get the first significant digit of the uncertainty
    if precision is None:
        precision = max(significant_digit(u) + 1 for u in uncertainty) or 3

    # Check if both uncertainties are equal up to precision
    if np.round(uncertainty[0], precision) == np.round(uncertainty[1], precision):
        return f"{value:.{precision}f} \u00b1 {uncertainty[0]:.{precision}f}"

    return f"{value:.{precision}f} +{uncertainty[1]:.{precision}f} / -{uncertainty[0]:.{precision}f}"


def data_uncertainties(data, method=None, data_median=None, homogeneous=True):
    """Compute the uncertainties of the median (or specified) values.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (float, optional): method of computing the method.
            If it is `None`, computes the standard deviation, otherwise it
            computes the corresponding confidence interval using ``np.percentile``.
            Defaults to ``None``.
        data_median (list or np.ndarray, optional): 1d array for computing the errors from the
            confidence interval. If ``None``, the median values are computed from ``data``.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size
            and returns ``np.ndarray``. Default is ``True``.

    Returns:
        np.ndarray: uncertainties of the data.
    """
    if method is None:
        return np.std(data, axis=1) if homogeneous else [np.std(row) for row in data]

    percentiles = [
        (100 - method) / 2,
        (100 + method) / 2,
    ]
    percentile_interval = np.percentile(data, percentiles, axis=1)

    uncertainties = np.abs(np.vstack([data_median, data_median]) - percentile_interval)

    return uncertainties


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


@dataclass
class RBData(Data):
    """The output of the acquisition function."""

    depths: list
    """Circuits depths."""
    uncertainties: Optional[float]
    """Parameters uncertainties."""
    seed: Optional[int]
    nshots: int
    """Number of shots."""
    niter: int
    """Number of iterations for each depth."""
    data: dict[Union[QubitId, QubitPairId], npt.NDArray[RBType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""
    npulses_per_clifford: float = 1.875
    """Number of pulses for an average clifford."""


@dataclass
class RB2QData(RBData):
    """The output of the acquisition function."""

    npulses_per_clifford: float = 8.6  # Assuming U3s and 1 pulse CZ
    """Number of pulses for an average clifford."""

    def extract_probabilities(self, qubits):
        """Extract the probabilities given (`qubit`, `qubit`)"""
        probs = []
        for depth in self.depths:
            data_list = np.array(self.data[qubits[0], qubits[1], depth].tolist())
            data_list = data_list.reshape((-1, self.nshots))
            probs.append(np.count_nonzero(1 - data_list, axis=1) / data_list.shape[1])
        return probs


@dataclass
class RB2QInterData(RB2QData):
    """The output of the acquisition function."""

    fidelity: dict[QubitPairId, list] = field(default_factory=dict)
    """The interleaved fidelity of this qubit."""


@dataclass
class StandardRBResult(Results):
    """Standard RB outputs."""

    fidelity: dict[QubitId, float]
    """The overall fidelity of this qubit."""
    pulse_fidelity: dict[QubitId, float]
    """The pulse fidelity of the gates acting on this qubit."""
    fit_parameters: dict[QubitId, list[float]]
    """Raw fitting parameters."""
    fit_uncertainties: dict[QubitId, list[float]]
    """Fitting parameters uncertainties."""
    error_bars: dict[QubitId, Optional[Union[float, list[float]]]] = field(
        default_factory=dict
    )
    """Error bars for y."""


def setup_data(
    params: Parameters,
    npulses_per_clifford: float,
    single_qubit: bool = True,
    interleave: Optional[str] = None,
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


def generate_indexed_circuits(
    params: Parameters,
    rb_gen: RBGenerator,
    targets,  # list[QubitId] or list[QubitPairId]
    inverse_layer: bool = True,
    interleave: Optional[str] = None,
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

    for depth in params.depths:
        for target in targets:
            for iteration in range(params.niter):
                circuit = layer_circuit(rb_gen, depth, target, interleave)
                if inverse_layer:
                    add_inverse_layer(circuit, rb_gen, inv_file)
                add_measurement_layer(circuit)

                index = CircuitIndex(qubit=target, depth=depth, iteration=iteration)
                indexed_circuits.append(IndexedCircuit(circuit=circuit, index=index))

    return indexed_circuits


def execute_indexed_circuits(
    indexed_circuits: list[IndexedCircuit],
    params: Parameters,
    platform: CalibrationPlatform,
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT,
) -> list[IndexedResult]:
    """Execute indexed circuits and return results paired with their indices.

    Args:
        indexed_circuits: List of IndexedCircuit objects to execute.
        targets: List of target qubit IDs.
        params: Experiment parameters.
        platform: CalibrationPlatform to execute on.

    Returns:
        List of IndexedResult objects with execution results paired with their indices.
    """

    qubit_maps = []
    circuits = []
    for indexed_circuit in indexed_circuits:
        qubit = indexed_circuit.index.qubit
        if isinstance(qubit, (list, tuple)):  # Multi-qubit
            qubit_maps.append(list(qubit))
        else:  # Single-qubit
            qubit_maps.append([qubit])
        circuits.append(indexed_circuit.circuit)

    transpiler = dummy_transpiler(platform)
    compiler = set_compiler(platform)

    transpiled_circuits = transpile_circuits(
        circuits,
        qubit_maps,
        platform,
        transpiler,
    )
    executed_results = execute_circuits(
        platform,
        compiler,
        transpiled_circuits,
        qubit_maps,
        nshots=params.nshots,
        averaging_mode=averaging_mode,
    )

    indexed_results = [
        IndexedResult(result=result, index=ic.index)
        for ic, result in zip(indexed_circuits, executed_results)
    ]

    return indexed_results


def rb_acquisition(
    params: Parameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
    inverse_layer: bool = True,
    interleave: str | None = None,
) -> RBData:
    """RB data acquisition function using explicit circuit indexing.

    Args:
        params: Experiment parameters including depths, niter, nshots, seed.
        platform: CalibrationPlatform to execute circuits on.
        targets: List of target qubit IDs.
        inverse_layer: Whether to add an inverse layer to circuits. Defaults to True.
        interleave: Interleaving pattern for circuits. Defaults to None.

    Returns:
        RBData: Validated RB data structure with results organized by (qubit, depth).
    """
    rb_gen = RBGenerator(params.seed)

    npulses_per_clifford = rb_gen.calculate_average_pulses()
    data = setup_data(
        params, npulses_per_clifford=npulses_per_clifford, single_qubit=True
    )

    indexed_circuits = generate_indexed_circuits(
        params=params,
        rb_gen=rb_gen,
        targets=targets,
        inverse_layer=inverse_layer,
        interleave=interleave,
    )

    indexed_results = execute_indexed_circuits(
        indexed_circuits=indexed_circuits,
        params=params,
        platform=platform,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # Create a dict of the form {(qubit, depth): list[result]}.
    # This marginalises over the iterations for a given (qubit, depth)
    grouped: defaultdict = defaultdict(list)
    for indexed_result in indexed_results:
        key = (indexed_result.index.qubit, indexed_result.index.depth)
        survival_counts = (
            indexed_result.result["0"] if inverse_layer else indexed_result.result["1"]
        )
        survival_prob = survival_counts / params.nshots
        grouped[key].append(survival_prob)

    for (qubit, depth), results in grouped.items():
        data.register_qubit(
            RBType,
            (qubit, depth),
            {"survival_probs": results},
        )

    return data


def twoq_rb_acquisition(
    params: Parameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
    inverse_layer: bool = True,
    interleave: str | None = None,
) -> Union[RB2QData, RB2QInterData]:
    """
    The data acquisition stage of two qubit Standard Randomized Benchmarking.

    Args:
        params (RB2QParameters): The parameters for the randomized benchmarking experiment.
        targets (list[QubitPairId]): The list of qubit pair IDs on which to perform the benchmarking.
        inverse_layer (bool, optional): Whether to add an inverse layer to the circuits. Defaults to True.
        interleave (str, optional): The type of interleaving to apply. Defaults to None.

    Returns:
        RB2QData: The acquired data for two qubit randomized benchmarking.
    """
    rb_gen = RBGenerator(params.seed, file=params.file)

    npulses_per_clifford = rb_gen.calculate_average_pulses()
    data = setup_data(
        params,
        npulses_per_clifford=npulses_per_clifford,
        single_qubit=False,
        interleave=interleave,
    )

    indexed_circuits = generate_indexed_circuits(
        params=params,
        rb_gen=rb_gen,
        targets=targets,
        inverse_layer=inverse_layer,
        interleave=interleave,
    )

    indexed_results = execute_indexed_circuits(
        indexed_circuits=indexed_circuits,
        params=params,
        platform=platform,
    )

    # Create a dict of the form {(qubit[0], qubit[1], depth): list[result]}.
    # This marginalises over the iterations for a given (qubit_pair, depth)
    grouped: defaultdict = defaultdict(list)
    for indexed_result in indexed_results:
        qubit_pair = indexed_result.index.qubit
        assert isinstance(qubit_pair, tuple)
        key = (qubit_pair[0], qubit_pair[1], indexed_result.index.depth)
        survival_counts = (
            indexed_result.result["00"]
            if inverse_layer
            else indexed_result.result["11"]
        )
        survival_prob = survival_counts / params.nshots
        grouped[key].append(survival_prob)

    for (qubit0, qubit1, depth), results in grouped.items():
        data.register_qubit(
            dtype=RBType,
            data_keys=(qubit0, qubit1, depth),
            data_dict={"survival_probs": results},
        )

    return data


def layer_circuit(
    rb_gen: RBGenerator,
    depth: int,
    target: Union[QubitId, QubitPairId],
    interleave: Optional[str] = None,
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
    elif isinstance(target, Tuple):  # Tuple for qubit pair
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


def fit(data, single_qubit: bool = True) -> StandardRBResult:
    """Takes data, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B."""

    if single_qubit:
        targets = data.qubits
        dimension = 2
    else:
        targets = data.pairs
        dimension = 2**2

    fidelity, pulse_fidelity = {}, {}
    popts, perrs = {}, {}
    error_barss = {}
    for target in targets:
        probs_array = np.array(
            [
                val["survival_probs"]
                for key, val in data.data.items()
                if (key[0] if single_qubit == 1 else key[:2]) == target
            ]
        )  # rows -> depths, cols -> iterations

        sigma = np.std(probs_array, axis=1)
        popt, perr = fit_exp1B_func(
            data.depths, np.mean(probs_array, axis=1), sigma=sigma, bounds=[0, 1]
        )

        # Compute the fidelities
        infidelity = (1 - popt[1]) * (dimension - 1) / dimension
        fidelity[target] = 1 - infidelity
        pulse_fidelity[target] = 1 - infidelity / data.npulses_per_clifford

        # conversion from np.array to list/tuple
        error_barss[target] = sigma.tolist()
        perrs[target] = perr
        popts[target] = popt

    return StandardRBResult(fidelity, pulse_fidelity, popts, perrs, error_barss)
