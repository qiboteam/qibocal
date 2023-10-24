from itertools import chain, product
from typing import Iterable, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qibo
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Qubits, Routine
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import add_measurement_layer, embed_circuit, layer_circuit
from .clifford_filtered_rb import CliffordRBResult
from .standard_rb import RBData, StandardRBParameters
from .utils import random_clifford_2q


# TODO: I assume this will change for 2q
def filter_function(samples_list, circuit_list) -> list:
    """Calculates the filtered signal for every crosstalk irrep.

    Every irrep has a projector characterized with a bitstring
    :math:`\\boldsymbol{\\lambda}\\in\\mathbb{F}_2^N` where :math:`N` is the
    number of qubits.
    The experimental outcome for each qubit is denoted as
    :math:`\\ket{i_k}` with :math:`i_k=0, 1` with :math:`d=2`.

    .. math::
        f_{\\boldsymbol{\\lambda}}(i,g)
        = \\frac{1}{2^{N-|\\boldsymbol{\\lambda}|}}
            \\sum_{\\mathbf b\\in\\mathbb F_2^N}
            (-1)^{|\\boldsymbol{\\lambda}\\wedge\\mathbf b|}
            \\frac{1}{d^N}\\left(\\prod_{k=1}^N(d|\\bra{i_k} U_{g_{(k)}}
            \\ket{0}|^2)^{\\lambda_k-\\lambda_kb_k}\\right)

    Args:
        samples_list (list or ndarray): list with lists of samples.
        circuit_list (Circuit): list of circuits used to produce the samples.

    Returns:
        datarow (dict):  Filtered signals are stored additionally.
    """

    # Extract amount of used qubits and used shots.
    nshots, nqubits = np.array(samples_list[0]).shape
    # For qubits the local dimension is 2.
    d = 2

    datarow = {f"irrep{kk}": [] for kk in range(d**nqubits)}

    for circuit, samples in zip(circuit_list, samples_list):
        # Fuse the gates for each qubit.
        fused_circuit = circuit.fuse(max_qubits=1)
        # Extract for each qubit the ideal state.
        # If depth = 0 there is only a measurement circuit and it does
        # not have an implemented matrix. Set the ideal states to ground states.
        if circuit.depth == 1:
            ideal_states = np.tile(np.array([1, 0]), nqubits).reshape(nqubits, 2)
        else:
            ideal_states = np.array(
                [fused_circuit.queue[k].matrix()[:, 0] for k in range(nqubits)]
            )
        # Go through every irrep.
        f_list = []
        for l in np.array(list(product([False, True], repeat=nqubits))):
            # Check if the trivial irrep is calculated
            if not sum(l):
                # In the end every value will be divided by ``nshots``.
                a = nshots
            else:
                # Get the supported ideal outcomes and samples
                # for this irreps projector.
                suppl = ideal_states[l]
                suppsamples = samples[:, l]
                a = 0
                # Go through all ``nshots`` samples
                for s in suppsamples:
                    # Go through all combinations of (0,1) on the support
                    # of lambda ``l``.
                    for b in np.array(list(product([False, True], repeat=sum(l)))):
                        # Calculate the sign depending on how many times the
                        # nontrivial projector was used.
                        # Take the product of all probabilities chosen by the
                        # experimental outcome which are supported by the
                        # inverse of b.
                        a += (-1) ** sum(b) * np.prod(
                            d * np.abs(suppl[~b][np.eye(2, dtype=bool)[s[~b]]]) ** 2
                        )
            # Normalize with inverse of effective measurement.
            f_list.append(a * (d + 1) ** sum(l) / d**nqubits)
        for kk in range(len(f_list)):
            datarow[f"irrep{kk}"].append(f_list[kk] / nshots)
    return datarow


# TODO: Can we assume you want to use 2q gates if you use QubitPairs and 1q gate if Qubits ?
def setup_scan(
    params: StandardRBParameters,
    qubits: list[tuple[QubitId]],
    nqubits: int,
    platform: Platform,
    **kwargs,
) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]):
            List of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    qubit_pairs = []
    for pair in qubits:
        qubit_pairs.append(platform.pairs[pair])

    qubit_ids = list(qubits) if isinstance(qubits, dict) else qubits
    qubits_circ_ids = np.arange(max(max(qubit_ids)) + 1)

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of ``depth``."""

        # This function is needed so that the inside of the layer_circuit function layer_gen()
        # can be called for each layer of the circuit, and it returns a random layer of
        # Clifford gates. Could also be a generator, it just has to be callable.
        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford_2q(qubit_pairs, params.seed)

        circuit = layer_circuit(layer_gen, depth, **kwargs)

        qubits_measure = list(chain(*qubit_ids))
        add_measurement_layer(circuit, qubits_measure)

        return embed_circuit(circuit, nqubits, qubits_circ_ids)

    return map(make_circuit, params.depths * params.niter)


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    qubits: Union[Qubits, list[QubitId]],
) -> RBData:
    """The data acquisition stage of Clifford Filtered Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]): List of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each exeriment in the scan.
    """

    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model:
        # FIXME: implement this check outside acquisition
        if platform and platform.name != "dummy":
            raise_error(
                NotImplementedError,
                f"Backend qibolab ({platform}) does not perform noise models simulation.",
            )
        elif platform:
            log.warning(
                (
                    "Backend qibolab (%s) does not perform noise models simulation. "
                    "Setting backend to ``NumpyBackend`` instead."
                ),
                platform.name,
            )
            qibo.set_backend("numpy")
            platform = None

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = platform.nqubits if platform else max(qubits) + 1
    scan = setup_scan(
        params, qubits, nqubits, platform, density_matrix=(noise_model is not None)
    )

    # 2. Execute the scan.
    data_list = []
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # Every executed circuit gets a row where the data is stored.
        depth = circuit.depth - 1 if circuit.depth > 0 else 0
        data_list.append({"depth": depth, "circuit": circuit})
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        samples = circuit.execute(nshots=params.nshots).samples()
        data_list[-1]["samples"] = samples

    clifford_rb_data = RBData(data_list)

    # TODO: They save noise model and noise params ...
    # Store the parameters to display them later.
    clifford_rb_data.attrs = params.__dict__

    # TODO: Can we assume you want to use 2q gates if you use QubitPairs and 1q gate if Qubits ?
    # element = next(iter(qubits.values()))
    # if isinstance(element, Qubit):
    #     clifford_rb_data.attrs.setdefault("qubits", qubits)
    # elif isinstance(element, QubitPair):
    #     clifford_rb_data.attrs.setdefault("qubit_pair", qubits)
    clifford_rb_data.attrs.setdefault("qubit_pair", qubits)

    return clifford_rb_data


def _fit(data: RBData) -> CliffordRBResult:
    filtered_data = pd.DataFrame()
    fit_results = pd.DataFrame()

    return CliffordRBResult(
        filtered_data,
        fit_results,
    )


def _plot(data: RBData, fit: CliffordRBResult, qubit) -> tuple[list[go.Figure], str]:
    result_fig = [go.Figure()]
    table_str = ""

    return result_fig, table_str


# Build the routine object which is used by qq.
clifford_filtered_rb_2q = Routine(_acquisition, _fit, _plot)
