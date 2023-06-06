from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from qibo.noise import NoiseModel

from qibocal.auto.operation import Routine
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels
from qibocal.protocols.characterization.randomized_benchmarking.circuit_tools import (
    add_inverse_layer,
    add_measurement_layer,
    embed_circuit,
    layer_circuit,
)
from qibocal.protocols.characterization.randomized_benchmarking.result import (
    DecayWithOffsetResult,
    plot_decay_result,
)
from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    extract_from_data,
    number_to_str,
    random_clifford,
)

from .data import RBData
from .params import RBParameters

PULSES_PER_CLIFFORD = 1.875


@dataclass
class StandardRBResult(DecayWithOffsetResult):
    """Inherits from `DecayWithOffsetResult`, a result class storing data and parameters
    of a single decay with statistics.

    Adds the method of calculating a fidelity out of the fitting parameters.
    TODO calculate SPAM errors with A and B
    TODO calculate the error of the fidelity

    """

    def __post_init__(self):
        super().__post_init__()
        self.resample_func = lambda data: resample_p0(
            data, self.meta_data.get("nshots", 1)
        )

    def calculate_fidelities(self):
        """Takes the fitting parameter of the decay and calculates a fidelity. Stores the
        primitive fidelity, the fidelity and the average gate error in an attribute dictionary.
        """
        infidelity = (1 - self.p) / 2
        self.fidelity_dict = {
            "fidelity": [1 - infidelity, self.perr / 2],
            "pi/2 fidelity": [
                1 - infidelity / PULSES_PER_CLIFFORD,
                self.perr / (PULSES_PER_CLIFFORD * 2),
            ],
        }


def samples_to_p0(samples_list):
    """Computes the probabilitiy of 0 from the list of samples.

    Args:
        samples_list (list or np.ndarray): 3d array with rows corresponding to circuits containing
            `nshots` number of lists with `nqubits` amount of `0` and `1`.
            e.g. `samples_list` for 1 circuit, 3 shots and 2 qubits looks like
            `[[[0, 0], [0, 1], [1, 0]]]`.

    Returns:
        list: list of probabilities corresponding to each row.
    """

    ground = np.array([0] * len(samples_list[0][0]))
    p0_list = []
    for samples in samples_list:
        p0_list.append(np.sum(np.product(samples == ground, axis=1)) / len(samples))
    return p0_list


def resample_p0(data, sample_size=100):
    """Preforms parametric resampling of shots with binomial distribution
        and returns a list of "corrected" probabilites.

    Args:
        data (list or np.ndarray): list of probabilities for the binomial distribution.
        nshots (int): sample size for one probability distribution.

    Returns:
        list: resampled probabilities.
    """
    # Parametrically sample the number of  shots with binomial distribution
    resampled_data = []
    for p in data:
        samples_corrected = np.random.binomial(n=1, p=1 - p, size=(sample_size, 1))
        resampled_data.append(samples_to_p0([samples_corrected])[0])
    return resampled_data


def setup_scan(params: RBParameters) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.
    ls
        Args:
            params (RBParameters): Parameters of the RB protocol.

        Returns:
            Iterable: The iterator of circuits.
    """

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of `depth`."""

        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford(len(params.qubits))

        circuit = layer_circuit(layer_gen, depth)
        add_inverse_layer(circuit)
        add_measurement_layer(circuit)
        return embed_circuit(circuit, params.nqubits, params.qubits)

    return map(make_circuit, params.depths * params.niter)


def execute(
    scan: Iterable,
    nshots: Optional[int] = None,
    noise_model: Optional[NoiseModel] = None,
) -> List[dict]:
    """Execute a given scan with the given number of shots and if its a simulation with the given
    noise model.

    Args:
        scan (Iterable): The ensemble of experiments (here circuits)
        nshots Optional[int]: Number of shots per circuit. Defaults to None.
        noise_model Optional[NoiseModel]: If its a simulation a noise model
            can be applied. Defaults to None.

    Returns:
        List[dict]: A list with one dictionary for each executed circuit where the data is stored.
    """

    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth - 2) if c.depth > 1 else 0
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    return data_list


def aggregate(data: RBData) -> StandardRBResult:
    """Takes a data frame, processes it and aggregates data in order to create
    a routine result object.

    Args:
        data (RBData): Actually a data frame from where the data is processed.

    Returns:
        StandardRBResult: The aggregated data.
    """

    # The signal is here the survival probability.
    data_agg = data.assign(signal=lambda x: samples_to_p0(x.samples.to_list()))
    return StandardRBResult(
        *extract_from_data(data_agg, "signal", "depth", list), meta_data=data.attrs
    )


def acquire(params: RBParameters, *args) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Put the acquired data in a standard rb data object.

    Args:
        params (RBParameters): All parameters in one object.

    Returns:
        RBData: _description_
    """

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    scan = setup_scan(params)
    # For simulations, a noise model can be added.
    noise_model = (
        getattr(noisemodels, params.noise_model)(*params.noise_params)
        if params.noise_model
        else None
    )
    # Execute the scan.
    data = execute(scan, params.nshots, noise_model)
    # Build the data object which will be returned and later saved.
    standardrb_data = RBData(data)
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def extract(data: RBData) -> StandardRBResult:
    """Takes a data frame and extracts the depths,
    average values of the survival probability and histogram

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    result = aggregate(data)
    result.fit()
    result.calculate_fidelities()
    return result


def plot(data: RBData, result: StandardRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """
    meta_data = deepcopy(result.meta_data)
    meta_data.pop("depths")
    if not meta_data["noise_model"]:
        meta_data.pop("noise_model")
        meta_data.pop("noise_params")
        meta_data.pop("nqubits")

    table_str = "".join([f" | {key}: {value}<br>" for key, value in meta_data.items()])

    table_str += "".join(
        f" | {key}: {number_to_str(*value)}<br>"
        for key, value in result.fidelity_dict.items()
    )
    fig = plot_decay_result(result)
    return [fig], table_str


# Build the routine object which is used by qq-auto.
standard_rb = Routine(acquire, extract, plot)
