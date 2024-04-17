from numbers import Number
from typing import Optional, Union

import numpy as np
from qibo import gates

from qibocal.protocols.characterization.utils import significant_digit

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


def random_clifford(random_index_gen):
    """Generates random Clifford operator.

    Args:
        qubits (int or list or ndarray): if ``int``, the number of qubits for the Clifford.
            If ``list`` or ``ndarray``, indexes of the qubits for the Clifford to act on.
        seed (int or ``numpy.random.Generator``, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Default is ``None``.

    Returns:
        (list of :class:`qibo.gates.Gate`): Random Clifford operator(s).
    """

    random_index = int(random_index_gen(SINGLE_QUBIT_CLIFFORDS))
    clifford_gate = SINGLE_QUBIT_CLIFFORDS[random_index](0)

    return clifford_gate, random_index


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
    This class generates random single qubit cliffords for randomized benchmarking.
    """

    def __init__(self, seed):
        self.seed = seed
        self.local_state = (
            np.random.default_rng(seed)
            if seed is None or isinstance(seed, int)
            else seed
        )

    def random_index(self, gate_list):
        """
        Generates a random index within the range of the given gate list.

        Parameters:
        - gate_list (list): Dict of gates.

        Returns:
        - int: Random index.
        """
        return self.local_state.integers(0, len(gate_list), 1)

    def layer_gen(self):
        """
        Returns:
        - Gate: Random single-qubit clifford .
        """
        return random_clifford(self.random_index)


def random_circuits(
    depth: int,
    targets: list[QubitId],
    niter,
    rb_gen,
    noise_model=None,
    inverse_layer=True,
) -> Iterable:
    """Returns single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        targets (list[QubitId]):
            list of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.
        inverse_layer (bool): If `True` a layer inverting the circuit is added.
            Default to `True`.

    Returns:
        Iterable: The iterator of circuits.
    """

    circuits = []
    indexes = defaultdict(list)
    for _ in range(niter):
        for target in targets:
            circuit, random_index = layer_circuit(rb_gen, depth, target)
            if inverse_layer:
                add_inverse_layer(circuit)
            add_measurement_layer(circuit)
            if noise_model is not None:
                circuit = noise_model.apply(circuit)
            circuits.append(circuit)
            indexes[target].append(random_index)

    return circuits, indexes


def rb_acquisition(
    params: Parameters,
    targets: list[QubitId],
    add_inverse: bool = True,
) -> RBData:
    """RB data acquisition function.

    Args:
        params (FilteredRBParameters): All parameters in one object.
        targets (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.

    """
    backend = GlobalBackend()
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
    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = len(targets)
    data = RBData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )

    circuits = []
    indexes = {}
    samples = []
    qubits_ids = targets
    rb_gen = RB_Generator(params.seed)
    for depth in params.depths:
        # TODO: This does not generate multi qubit circuits
        circuits_depth, random_indexes = random_circuits(
            depth,
            qubits_ids,
            params.niter,
            rb_gen,
            noise_model,
            add_inverse,
        )
        circuits.extend(circuits_depth)
        for qubit in random_indexes.keys():
            indexes[(qubit, depth)] = random_indexes[qubit]
    # Execute the circuits
    if params.unrolling:
        executed_circuits = backend.execute_circuits(circuits, nshots=params.nshots)
    else:
        executed_circuits = [
            backend.execute_circuit(circuit, nshots=params.nshots)
            for circuit in circuits
        ]

    for circ in executed_circuits:
        samples.extend(circ.samples())
    samples = np.reshape(samples, (-1, nqubits, params.nshots))

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

    return data
