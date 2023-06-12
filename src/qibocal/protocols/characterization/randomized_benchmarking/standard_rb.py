from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qibo
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import (
    add_inverse_layer,
    add_measurement_layer,
    embed_circuit,
    layer_circuit,
)
from .fitting import exp1B_func, fit_exp1B_func
from .utils import extract_from_data, random_clifford

NPULSES_PER_CLIFFORD = 1.875


class DepthsDict(TypedDict):
    """Dictionary used to build a list of depths as `range(start, stop, step)`."""

    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    """The amount of qubits on the chip """
    depths: Union[list, DepthsDict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
    noise_params: Optional[list] = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


class RBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to_csv(self, path):
        """Overwrite because qibocal action builder calls this function with a directory."""
        super().to_json(f"{path}/{self.__class__.__name__}.csv")


@dataclass
class StandardRBResult(Results):
    """Standard RB outputs."""

    fidelity: float
    """The overall fidelity of this qubit."""
    pulse_fidelity: float
    """The pulse fidelity of the gates acting on this qubit."""
    fitting_parameters: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """Raw fitting parameters."""


def setup_scan(params: StandardRBParameters, qubits) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.

    Returns:
        Iterable: The iterator of circuits.
    """
    qubit_ids = list(qubits) if isinstance(qubits, dict) else qubits

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of `depth`."""

        # This function is needed so that the inside of the layer_circuit function layer_gen()
        # can be called for each layer of the circuit, and it returns a random layer of
        # Clifford gates. Could also be a generator, it just has to be callable.
        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford(len(qubit_ids))

        circuit = layer_circuit(layer_gen, depth)
        add_inverse_layer(circuit)
        add_measurement_layer(circuit)
        return embed_circuit(circuit, params.nqubits, qubit_ids)

    return map(make_circuit, params.depths * params.niter)


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    qubits: Union[Qubits, List[QubitId]],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits: List of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model:
        # FIXME implement this check outside acquisition
        if platform and platform.name != "dummy":
            raise_error(
                NotImplementedError,
                f"Backend qibolab ({platform}) does not perform noise models simulation.",
            )
        elif platform:
            log.warning(
                (
                    "Backend qibolab (%s) does not perform noise models simulation. "
                    "Setting backend to `NumpyBackend` instead."
                ),
                platform.name,
            )
            qibo.set_backend("numpy")

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    scan = setup_scan(params, qubits)

    # 2. Execute the scan.
    data_list = []
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (circuit.depth - 2) if circuit.depth > 1 else 0
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        samples = circuit.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    # Build the data object which will be returned and later saved.
    data = pd.DataFrame(data_list)

    def p0s(samples_list):
        ground = np.array([0] * len(samples_list[0][0]))
        my_p0s = []
        for samples in samples_list:
            my_p0s.append(np.sum(np.product(samples == ground, axis=1)) / len(samples))
        return my_p0s

    # The signal here is the survival probability.
    standardrb_data = RBData(data.assign(signal=lambda x: p0s(x.samples.to_list())))
    # Store the parameters to display them later.
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    x, y = extract_from_data(data, "signal", "depth", "mean")
    popt, perr = fit_exp1B_func(x, y)
    infidelity = (1 - popt[0]) / 2
    fidelity = np.round(1 - infidelity, 4)
    pulse_fidelity = np.round(1 - infidelity / NPULSES_PER_CLIFFORD, 4)
    return StandardRBResult(fidelity, pulse_fidelity, (popt, perr))


def _plot(data: RBData, result: StandardRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    x, y = extract_from_data(data, "signal", "depth", list)
    popt, perr = result.fitting_parameters
    label = (
        "Fit: y=Ap^x+B<br>"
        f"A: {popt[0]:.3f}\u00B1{perr[0]:.3f}<br>"
        f"p: {popt[1]:.3f}\u00B1{perr[1]:.3f}<br>"
        f"B: {popt[2]:.3f}\u00B1{perr[2]:.3f}"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.repeat(x, len(y[0])),
            y=np.array([np.array(y_row) for y_row in y]).flatten(),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="itertarions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[np.mean(y_row) for y_row in y],
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    x_fit = np.linspace(min(x), max(x), len(x) * 20)
    y_fit = exp1B_func(x_fit, *popt)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=label,
            line=go.scatter.Line(dash="dot"),
        )
    )

    meta_data = deepcopy(data.attrs)
    meta_data.pop("depths")
    if not meta_data["noise_model"]:
        meta_data.pop("noise_model")
        meta_data.pop("noise_params")
        meta_data.pop("nqubits")

    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in {
                **meta_data,
                "fidelity": result.fidelity,
                "pulse_fidelity": result.pulse_fidelity,
            }.items()
        ]
    )
    return [fig], table_str


# Build the routine object which is used by qq-auto.
standard_rb = Routine(_acquisition, _fit, _plot)
