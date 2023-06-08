from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo.config import raise_error
from qibo.noise import NoiseModel

from qibocal.auto.operation import Parameters, Results, Routine
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import (
    add_inverse_layer,
    add_measurement_layer,
    embed_circuit,
    layer_circuit,
)
from .fitting import bootstrap, exp1B_func, fit_exp1B_func
from .utils import (
    data_mean_errors,
    extract_from_data,
    number_to_str,
    random_clifford,
    resample_p0,
    samples_to_p0,
)

NPULSES_PER_CLIFFORD = 1.875


class DepthsDict(TypedDict):
    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    """The amount of qubits on the chip """
    qubits: list
    """A list of indices which qubit(s) should be benchmarked """
    depths: Union[list, DepthsDict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    uncertainties: Union[str, float] = 0.95
    """Method of computing the error bars and uncertainties of the data. If ``None``, does not
    compute the errors. If ``"std"``, computes the standard deviation. If a value is of type ``float``
    between 0 and 1, computes the corresponding confidence interval. Defaults to ``0.95``."""
    n_bootstrap: int = 100
    """Number of bootstrap iterations for the fit uncertainties and error bars.
    If ``0``, gets the fit uncertainties from the fitting function and the error bars
    from the distribution of the measurements. Defaults to ``1``."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
    noise_params: list = field(default_factory=list)
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
        """Overwrite this method because qibocal action builder call this function with a directory."""
        super().to_csv(f"{path}/{self.__class__.__name__}.csv")


@dataclass
class StandardRBResult(Results):
    fidelity: float
    """The overall fidelity of this qubit."""
    pulse_fidelity: float
    """The pulse fidelity of the gates acting on this qubit."""
    fitting_parameters: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """Raw fitting parameters."""
    error_y: Optional[Union[float, List[float], np.ndarray]] = None
    """Error bars for y."""


def setup_scan(params: StandardRBParameters) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.

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


def _acquisition(params: StandardRBParameters, platform) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.

    Returns:
        RBData: The depths, samples and ground state probability of each exeriment in the scan.
    """

    if platform and params.noise_model:
        raise_error(
            TypeError,
            f"Platform {platform} is for hardware, you need a backend for simulation",
        )

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    scan = setup_scan(params)
    # For simulations, a noise model can be added.
    noise_model = (
        getattr(noisemodels, params.noise_model)(*params.noise_params)
        if params.noise_model
        else None
    )
    # TODO extract the noise parameters from the build noise model and add them to params object.
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
    data = pd.DataFrame(data)

    # The signal here is the survival probability.
    standardrb_data = RBData(
        data.assign(signal=lambda x: samples_to_p0(x.samples.to_list()))
    )
    # Store the paramters to display them later.
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
    # Extract depths and probabilities
    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    y = [np.mean(y_row) for y_row in y_scatter]

    # Extract fitting and bootstrap parameters if given
    uncertainties = data.attrs.get("uncertainties", None)
    n_bootstrap = data.attrs.get("n_bootstrap", 0)
    init_sigma = data.attrs.get("sigma", None)

    # Perform bootstrap resampling
    y_estimates, popt_estimates = bootstrap(
        x,
        y_scatter,
        uncertainties,
        n_bootstrap,
        lambda data: resample_p0(data, data.attrs.get("nshots", 1)),
        fit_exp1B_func,
    )

    # Fit the initial data
    sigma = (
        data_mean_errors(y_estimates, uncertainties, symmetric=True)
        if init_sigma is None
        else init_sigma
    )
    popt, perr = fit_exp1B_func(x, y, sigma=sigma)

    # Compute fitting errors
    if len(popt_estimates):
        perr = data_mean_errors(popt_estimates, uncertainties)
        perr = perr.T if perr is not None else (0,) * len(perr)

    # Compute y data errors
    error_y = data_mean_errors(y_estimates, uncertainties, symmetric=False)

    infidelity = (1 - popt[0]) / 2
    fidelity = 1 - infidelity
    pulse_fidelity = 1 - infidelity / NPULSES_PER_CLIFFORD
    return StandardRBResult(fidelity, pulse_fidelity, (popt, perr), error_y)


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

    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    y = np.mean(y_scatter, axis=1)
    popt, perr = result.fitting_parameters
    label = "Fit: y=Ap^x<br>A: {}<br>p: {}<br>B: {}".format(
        number_to_str(popt[0], perr[0]),
        number_to_str(popt[1], perr[1]),
        number_to_str(popt[2], perr[2]),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.repeat(x, len(y_scatter[0])),
            y=np.array([np.array(y_row) for y_row in y_scatter]).flatten(),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="itertarions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    # If result.error_y is given, create a dictionary for the error bars
    error_y_dict = None
    if result.error_y is not None:
        # Constant error bars
        if isinstance(result.error_y, Iterable) is False:
            error_y_dict = {"type": "constant", "value": result.error_y}
        # Symmetric error bars
        elif isinstance(result.error_y[0], Iterable) is False:
            error_y_dict = {"type": "data", "array": result.error_y}
        # Asymmetric error bars
        else:
            error_y_dict = {
                "type": "data",
                "symmetric": False,
                "array": result.error_y[1],
                "arrayminus": result.error_y[0],
            }
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=error_y_dict,
                line={"color": "#aa6464"},
                mode="markers",
                name="error bars",
            )
        )
    x_fit = np.linspace(min(x), max(x), len(x) * 20)
    y_fit = exp1B_func(x_fit, *popt)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=label,
            line=go.scatter.Line(dash="dot", color="#00cc96"),
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
                "fidelity": number_to_str(result.fidelity, perr[1] / 2),
                "pulse_fidelity": number_to_str(
                    result.pulse_fidelity, perr[1] / (2 * NPULSES_PER_CLIFFORD)
                ),
            }.items()
        ]
    )
    return [fig], table_str


# Build the routine object which is used by qq-auto.
standard_rb = Routine(_acquisition, _fit, _plot)
