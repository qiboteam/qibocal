import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Number
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from qibo import gates
from qibo.backends import get_backend
from qibo.config import raise_error
from qibo.models import Circuit
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Parameters, Results
from qibocal.auto.transpile import (
    dummy_transpiler,
    execute_transpiled_circuit,
    execute_transpiled_circuits,
)
from qibocal.config import raise_error
from qibocal.protocols.randomized_benchmarking import noisemodels
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
        ("samples", np.int32),
    ]
)
"""Custom dtype for RB."""


def random_clifford(random_index_gen):
    """Generates random Clifford operator."""

    random_index = int(random_index_gen(SINGLE_QUBIT_CLIFFORDS))
    clifford_gate = SINGLE_QUBIT_CLIFFORDS[random_index](0)

    return clifford_gate, random_index


def random_2q_clifford(random_index_gen, two_qubit_cliffords):
    """Generates random two qubit Clifford operator."""

    random_index = int(random_index_gen(two_qubit_cliffords))
    clifford = two_qubit_cliffords[str(random_index)]
    clifford_gate = clifford2gates(clifford)

    return clifford_gate, random_index


def random_circuits(
    depth: int,
    targets: list[Union[QubitId, QubitPairId]],
    niter,
    rb_gen,
    noise_model=None,
    inverse_layer=True,
    single_qubit=True,
    file_inv=pathlib.Path(),
    interleave=None,
) -> Iterable:
    """Returns random (self-inverting) Clifford circuits."""

    circuits = []
    indexes = defaultdict(list)
    for _ in range(niter):
        for target in targets:
            circuit, random_index = layer_circuit(rb_gen, depth, target, interleave)
            if inverse_layer:
                add_inverse_layer(circuit, rb_gen, single_qubit, file_inv)
            add_measurement_layer(circuit)
            if noise_model is not None:
                circuit = noise_model.apply(circuit)
            circuits.append(circuit)
            indexes[target].append(random_index)

    return circuits, indexes


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
        return f"{value:.{precision}f} \u00B1 {uncertainty:.{precision}f}"

    # If any uncertainty is None, return the value with precision
    if any(u is None for u in uncertainty):
        return f"{value:.{precision if precision is not None else 3}f}"

    # If precision is None, get the first significant digit of the uncertainty
    if precision is None:
        precision = max(significant_digit(u) + 1 for u in uncertainty) or 3

    # Check if both uncertainties are equal up to precision
    if np.round(uncertainty[0], precision) == np.round(uncertainty[1], precision):
        return f"{value:.{precision}f} \u00B1 {uncertainty[0]:.{precision}f}"

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


class RB_Generator:
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
        return self.local_state.integers(0, len(gate_dict), 1)

    def layer_gen_single_qubit(self):
        """Generates a random single-qubit clifford gate."""
        return random_clifford(self.random_index)

    def layer_gen_two_qubit(self):
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
    circuits: dict[Union[QubitId, QubitPairId], list[list[int]]] = field(
        default_factory=dict
    )
    """Clifford gate indexes executed."""
    npulses_per_clifford: float = 1.875
    """Number of pulses for an average clifford."""

    def extract_probabilities(self, qubit):
        """Extract the probabilities given `qubit`"""
        probs = []
        for depth in self.depths:
            data_list = np.array(self.data[qubit, depth].tolist())
            data_list = data_list.reshape((-1, self.nshots))
            probs.append(np.count_nonzero(1 - data_list, axis=1) / data_list.shape[1])
        return probs


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


def setup(
    params: Parameters,
    platform: Platform,
    single_qubit: bool = True,
    interleave: Optional[str] = None,
):
    """
    Set up the randomized benchmarking experiment backend, noise model and data class.

    Args:
        params (Parameters): The parameters for the experiment.
        single_qubit (bool, optional): Flag indicating whether the experiment is for a single qubit or two qubits. Defaults to True.
        interleave: (str, optional): The type of interleaving to apply. Defaults to None.

    Returns:
        tuple: A tuple containing the experiment data, noise model, and backend.
    """

    backend = get_backend()
    backend.platform = platform
    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model is not None:
        if backend.name == "qibolab":
            raise_error(
                ValueError,
                "Backend qibolab (%s) does not perform noise models simulation. ",
            )

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params.tolist()
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
    )

    return data, noise_model, backend


def get_circuits(
    params, targets, add_inverse_layer, interleave, noise_model, single_qubit=True
):
    """
    Generate randomized benchmarking circuits.

    Args:
        params (Parameters): Experiment parameters.
        targets (list): List of target qubit IDs.
        add_inverse_layer (bool): Flag indicating whether to add an inverse layer to the circuits.
        interleave (str): String indicating whether to interleave the circuits with the given gate.
        noise_model (str): Noise model string.
        single_qubit (bool, optional): Flag indicating whether to generate single qubit circuits.

    Returns:
        tuple: A tuple containing the generated circuits, indexes, and the number of pulses per Clifford.
            - circuits (list): List of generated circuits.
            - indexes (dict): Dictionary containing the random indexes for each qubit and depth.
            - npulses_per_clifford (float): Number of pulses per Clifford.

    """
    circuits = []
    indexes = {}
    qubits_ids = targets
    rb_gen = (
        RB_Generator(params.seed)
        if single_qubit
        else RB_Generator(params.seed, params.file)
    )
    npulses_per_clifford = rb_gen.calculate_average_pulses()
    inv_file = params.file_inv if not single_qubit else None
    for depth in params.depths:
        # TODO: This does not generate multi qubit circuits
        circuits_depth, random_indexes = random_circuits(
            depth,
            qubits_ids,
            params.niter,
            rb_gen,
            noise_model,
            add_inverse_layer,
            single_qubit,
            inv_file,
            interleave,
        )

        circuits.extend(circuits_depth)
        if single_qubit:
            for qubit in random_indexes:
                indexes[(qubit, depth)] = random_indexes[qubit]
        else:
            for qubit in random_indexes:
                indexes[(qubit[0], qubit[1], depth)] = random_indexes[qubit]

    return circuits, indexes, npulses_per_clifford


def execute_circuits(circuits, targets, params, backend, single_qubit=True):
    """
    Executes a list of circuits on a given backend.

    Args:
        circuits (list): List of quantum circuits to be executed.
        targets (list): List of target qubits for each circuit.
        params (Parameters): Experiment parameters.
        backend (object): Backend to execute the circuits on.
        single_qubit (bool): Flag indicating whether the circuits are single-qubit or multi-qubit.

    Returns:
        list: List of executed circuits.

    """
    # Execute the circuits
    transpiler = dummy_transpiler(backend)
    qubit_maps = (
        [[i] for i in targets] * (len(params.depths) * params.niter)
        if single_qubit
        else [list(i) for i in targets] * (len(params.depths) * params.niter)
    )
    if params.unrolling:
        _, executed_circuits = execute_transpiled_circuits(
            circuits,
            qubit_maps=qubit_maps,
            backend=backend,
            nshots=params.nshots,
            transpiler=transpiler,
        )
    else:
        executed_circuits = [
            execute_transpiled_circuit(
                circuit,
                qubit_map=qubit_map,
                backend=backend,
                nshots=params.nshots,
                transpiler=transpiler,
            )[1]
            for circuit, qubit_map in zip(circuits, qubit_maps)
        ]
    return executed_circuits


def rb_acquisition(
    params: Parameters,
    platform: Platform,
    targets: list[QubitId],
    add_inverse_layer: bool = True,
    interleave: str = None,
) -> RBData:
    """RB data acquisition function.

    This function performs data acquisition for randomized benchmarking experiments.

    Args:
        params (RBParameters): All parameters in one object.
        targets (dict[int, Union[str, int]] or list[Union[str, int]]): List of qubits the experiment is executed on.
        add_inverse_layer (bool, optional): Whether to add an inverse layer to the circuits. Defaults to True.
        interleave (str, optional): Interleaving pattern for the circuits. Defaults to None.

    Returns:
        RBData: The depths, samples, and ground state probability of each experiment in the scan.
    """
    data, noise_model, backend = setup(params, platform, single_qubit=True)
    circuits, indexes, npulses_per_clifford = get_circuits(
        params, targets, add_inverse_layer, interleave, noise_model, single_qubit=True
    )
    executed_circuits = execute_circuits(circuits, targets, params, backend)

    samples = []
    for circ in executed_circuits:
        samples.extend(circ.samples())
    samples = np.reshape(samples, (-1, len(targets), params.nshots))

    for i, depth in enumerate(params.depths):
        index = (i * params.niter, (i + 1) * params.niter)
        for nqubit, qubit_id in enumerate(targets):
            data.register_qubit(
                RBType,
                (qubit_id, depth),
                dict(
                    samples=samples[index[0] : index[1]][:, nqubit],
                ),
            )
    data.circuits = indexes
    data.npulses_per_clifford = npulses_per_clifford

    return data


def twoq_rb_acquisition(
    params: Parameters,
    platform: Platform,
    targets: list[QubitPairId],
    add_inverse_layer: bool = True,
    interleave: str = None,
) -> Union[RB2QData, RB2QInterData]:
    """
    The data acquisition stage of two qubit Standard Randomized Benchmarking.

    Args:
        params (RB2QParameters): The parameters for the randomized benchmarking experiment.
        targets (list[QubitPairId]): The list of qubit pair IDs on which to perform the benchmarking.
        add_inverse_layer (bool, optional): Whether to add an inverse layer to the circuits. Defaults to True.
        interleave (str, optional): The type of interleaving to apply. Defaults to None.

    Returns:
        RB2QData: The acquired data for two qubit randomized benchmarking.
    """

    data, noise_model, backend = setup(params, platform, single_qubit=False)
    circuits, indexes, npulses_per_clifford = get_circuits(
        params, targets, add_inverse_layer, interleave, noise_model, single_qubit=False
    )
    executed_circuits = execute_circuits(
        circuits, targets, params, backend, single_qubit=False
    )

    samples = []
    for circ in executed_circuits:
        # Post process [0,0] to 0 and [1,0], [0,1], [1,1] to 1
        converted_samples = (
            1 - np.all(circ.samples() == [0], axis=1).astype(int)
        ).tolist()
        samples.extend(converted_samples)
    samples = np.reshape(samples, (-1, len(targets), params.nshots))

    for i, depth in enumerate(params.depths):
        index = (i * params.niter, (i + 1) * params.niter)
        for nqubit, qubit_id in enumerate(targets):
            data.register_qubit(
                RBType,
                (qubit_id[0], qubit_id[1], depth),
                dict(
                    samples=samples[index[0] : index[1]][:, nqubit],
                ),
            )
    data.circuits = indexes
    data.npulses_per_clifford = npulses_per_clifford

    return data


def layer_circuit(
    rb_gen: Callable, depth: int, target, interleave: str = None
) -> tuple[Circuit, list]:
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
    random_indexes = []
    if isinstance(target, (str, int)):
        nqubits = 1
        rb_gen_layer = rb_gen.layer_gen_single_qubit
    elif isinstance(target, Tuple):  # Tuple for qubit pair
        nqubits = 2
        rb_gen_layer = rb_gen.layer_gen_two_qubit
    # Build each layer, there will be depth many in the final circuit.
    for _ in range(depth):
        # Generate a layer.
        new_layer, random_index = rb_gen_layer()
        random_indexes.append(random_index)
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
                random_indexes.append("13")

        if full_circuit is None:  # instantiate in first loop
            full_circuit = Circuit(new_circuit.nqubits)
        full_circuit = full_circuit + new_circuit
    return full_circuit, random_indexes


def add_inverse_layer(
    circuit: Circuit, rb_gen: RB_Generator, single_qubit=True, file_inv=pathlib.Path()
):
    """Adds an inverse gate/inverse gates at the end of a circuit (in place).

    Args:
        circuit (Circuit): circuit
    """
    if single_qubit:
        if circuit.depth > 0:
            circuit.add(
                gates.Unitary(circuit.unitary(), *range(circuit.nqubits)).dagger()
            )
    else:
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


def add_measurement_layer(circuit: Circuit):
    """Adds a measurement layer at the end of the circuit.

    Args:
        circuit (Circuit): Measurement gates added in place to end of this circuit.
    """

    circuit.add(gates.M(*range(circuit.nqubits)))


def fit(qubits, data):
    """Takes data, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B."""

    fidelity, pulse_fidelity = {}, {}
    popts, perrs = {}, {}
    error_barss = {}
    for qubit in qubits:
        # Extract depths and probabilities
        x = data.depths
        probs = data.extract_probabilities(qubit)
        samples_mean = np.mean(probs, axis=1)
        # TODO: Should we use the median or the mean?
        median = np.median(probs, axis=1)

        error_bars = data_uncertainties(
            probs,
            method=data.uncertainties,
            data_median=median,
        )

        sigma = (
            np.max(error_bars, axis=0) if data.uncertainties is not None else error_bars
        )

        popt, perr = fit_exp1B_func(x, samples_mean, sigma=sigma, bounds=[0, 1])
        # Compute the fidelities
        infidelity = (1 - popt[1]) / 2
        fidelity[qubit] = 1 - infidelity
        pulse_fidelity[qubit] = 1 - infidelity / data.npulses_per_clifford

        # conversion from np.array to list/tuple
        error_bars = error_bars.tolist()
        error_barss[qubit] = error_bars
        perrs[qubit] = perr
        popts[qubit] = popt

    return StandardRBResult(fidelity, pulse_fidelity, popts, perrs, error_barss)
