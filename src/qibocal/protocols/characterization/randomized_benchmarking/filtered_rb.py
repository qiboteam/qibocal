from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibo.backends import GlobalBackend
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Routine
from qibocal.config import raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .standard_rb import (
    FilteredRBResult,
    RB_Generator,
    RBData,
    RBType,
    StandardRBResult,
    random_circuits,
)


@dataclass
class FilteredRBParameters(StandardRBParamters):
    """Filtered Randomized Benchmarking runcard inputs."""


@dataclass
class FilteredRBResult(StandardRBResult):
    """Filtered RB outputs."""


def _acquisition(
    params: FilteredRBParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RBData:
    """The data acquisition stage of Filtered Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a filtered rb data object with it.

    Args:
        params (FilteredRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

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
            depth, qubits_ids, params.niter, rb_gen, noise_model, inverse_layer=False
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


def _fit(data: RBData) -> FilteredRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        FilteredRBResult: Aggregated and processed data.
    """
    pass


def _plot(
    data: RBData, fit: FilteredRBResult, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        fit (FilteredRBResult): Is called for the plot.
        target (_type_): Not used yet.

    Returns:
        tuple[list[go.Figure], str]:
    """
    pass


filtered_rb = Routine(_acquisition, _fit, _plot)
