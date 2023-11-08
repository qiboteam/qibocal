from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
import plotly.graph_objects as go
import qibo
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.bootstrap import data_uncertainties
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from ..utils import table_dict, table_html
from .circuit_tools import (
    add_inverse_layer,
    add_measurement_layer,
    embed_circuit,
    layer_circuit,
)
from .data import RBData, RBType
from .fitting import exp1B_func, fit_exp1B_func
from .utils import number_to_str, random_clifford

NPULSES_PER_CLIFFORD = 1.875


class Depthsdict(TypedDict):
    """dictionary used to build a list of depths as ``range(start, stop, step)``."""

    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    depths: Union[list, Depthsdict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    uncertainties: Union[str, float] = 95
    """Method of computing the error bars of the signal and uncertainties of the fit. If ``None``,
    does not compute them. If ``"std"``, computes the standard deviation. If ``float`` or ``int``
    between 0 and 100, computes the corresponding confidence interval. Defaults to ``95``."""
    n_bootstrap: int = 100
    """Number of bootstrap iterations for the fit uncertainties and error bars.
    If ``0``, gets the fit uncertainties from the fitting function and the error bars
    from the distribution of the measurements. Defaults to ``100``."""
    seed: Optional[int] = None
    """A fixed seed to initialize ``np.random.Generator``. If ``None``, uses a random seed.
    Defaults is ``None``."""
    parallel: bool = False
    """Marginalize data to get several RBs from a big circuit"""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
    noise_params: Optional[list] = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""
    nshots: int = 1
    """Just to add the default value"""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


@dataclass
class StandardRBResult(Results):
    """Standard RB outputs."""

    fidelity: dict[QubitId, float]
    """The overall fidelity of this qubit."""
    pulse_fidelity: dict[QubitId, float]
    """The pulse fidelity of the gates acting on this qubit."""
    fit_parameters: dict[QubitId, tuple[float, float, float]]
    """Raw fitting parameters."""
    fit_uncertainties: dict[QubitId, tuple[float, float, float]]
    """Fitting parameters uncertainties."""
    error_bars: dict[QubitId, Optional[Union[float, list[float]]]] = None
    """Error bars for y."""


def samples_to_p0(samples_list):
    """Computes the probabilitiy of 0 from the list of samples.

    Args:
        samples_list (list or np.ndarray): 3d array with ``ncircuits`` rows containing
            ``nshots`` lists with ``nqubits`` amount of ``0`` and ``1`` samples.
            e.g. ``samples_list`` for 1 circuit, 3 shots and 2 qubits looks like
            ``[[[0, 0], [0, 1], [1, 0]]]`` and ``p0=1/3``.

    Returns:
        list: list of probabilities corresponding to each row.
    """

    ground = np.array([0] * len(samples_list[0][0]))
    return np.count_nonzero((samples_list == ground).all(axis=2), axis=1) / len(
        samples_list[0]
    )


def resample_p0(data, sample_size=100, homogeneous: bool = True):
    """Preforms parametric resampling of shots with binomial distribution
        and returns a list of "corrected" probabilites.

    Args:
        data (list or np.ndarray): list of probabilities for the binomial distribution.
        nshots (int): sample size for one probability distribution.

    Returns:
        list: resampled probabilities.
    """
    if homogeneous:
        return np.apply_along_axis(
            lambda p: samples_to_p0(
                np.random.binomial(n=1, p=1 - p, size=(1, sample_size, len(p))).T
            ),
            0,
            data,
        )

    resampled_data = []
    for row in data:
        resampled_data.append([])
        for p in row:
            samples_corrected = np.random.binomial(
                n=1, p=1 - p, size=(1, sample_size, *p.shape)
            ).T
            resampled_data[-1].append(samples_to_p0(samples_corrected))
    return resampled_data


def setup_scan(
    params: StandardRBParameters, qubits: Union[Qubits, list[QubitId]], nqubits: int
) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]):
            list of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    qubit_ids = list(qubits) if isinstance(qubits, dict) else qubits

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of ``depth``."""

        # This function is needed so that the inside of the layer_circuit function layer_gen()
        # can be called for each layer of the circuit, and it returns a random layer of
        # Clifford gates. Could also be a generator, it just has to be callable.
        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford(len(qubit_ids), params.seed)

        circuit = layer_circuit(layer_gen, depth)
        add_inverse_layer(circuit)
        add_measurement_layer(circuit)

        return embed_circuit(circuit, nqubits, qubit_ids)

    return map(make_circuit, params.depths * params.niter)


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    qubits: Union[Qubits, list[QubitId]],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

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
                    "Setting backend to ``NumpyBackend`` instead."
                ),
                platform.name,
            )
            qibo.set_backend("numpy")
            platform = None

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = list(noise_model.params)

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = platform.nqubits if platform else max(qubits) + 1
    scan = setup_scan(params, qubits, nqubits)

    # 2. Execute the scan.
    data = RBData(params=params.__dict__)
    samples = defaultdict(list)
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (circuit.depth - 2) if circuit.depth > 1 else 0
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        sample = circuit.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        # TODO: Try to get register qubit here
        samples[depth].append(sample.tolist())

    new_samples = defaultdict(list)
    for qn, q in enumerate(qubits):
        for d in params.depths:
            for i in range(params.niter):
                for n in range(params.nshots):
                    new_samples[q, d].append(samples[d][i][n][qn])

    for i, qubit in enumerate(qubits):
        for depth in params.depths:
            data.register_qubit(
                RBType,
                (qubit, depth),
                dict(
                    samples=new_samples[qubit, depth],
                ),
            )

    #     # signals = defaultdict(list)
    #     # for key, value in samples.items():
    #     #     if params.parallel:
    #     #         signals[key] = samples_to_p0s(list(chain.from_iterable(value)))
    #     #     else:
    #     #         # TODO: Fix this
    #     #         signals[key] = samples_to_p0(list(chain.from_iterable(value)))

    # else:
    #     # The signal here is the survival probability.
    #     standardrb_data = RBData(
    #         data.assign(signal=lambda x: samples_to_p0(x.samples.to_list()))
    #     )

    # Store the parameters to display them later.
    data.params = params.__dict__

    return data


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    qubits = data.qubits

    fidelity = {}
    pulse_fidelity = {}
    popts = {}
    perrs = {}
    error_barss = {}

    for qubit in qubits:
        # Extract depths and probabilities
        x = data.params["depths"]
        y_scatter = data.samples_to_p0s(qubit, x)
        # TODO: Remove this extra list needed to work
        y_scatter = [y_scatter]

        """This is when you sample a depth more than once"""
        # homogeneous = all(len(y_scatter[0]) == len(row) for row in y_scatter)
        homogeneous = True

        # Extract fitting and bootstrap parameters if given
        uncertainties = data.params["uncertainties"]
        n_bootstrap = data.params["n_bootstrap"]

        popt_estimates = []

        # FIXME: Let's disable it for now
        n_bootstrap = None
        # if uncertainties and n_bootstrap:
        #     # Non-parametric bootstrap resampling
        #     bootstrap_y = bootstrap(
        #         y_scatter,
        #         n_bootstrap,
        #         homogeneous=homogeneous,
        #         seed=data.params["seed"],
        #     )

        #     # Parametric bootstrap resampling of "corrected" probabilites from binomial distribution
        #     bootstrap_y = resample_p0(
        #         bootstrap_y, data.params["nshots"], homogeneous=homogeneous
        #     )

        #     # Compute y and popt estimates for each bootstrap iteration
        #     y_estimates = (
        #         np.mean(bootstrap_y, axis=1)
        #         if homogeneous
        #         else [np.mean(y_iter, axis=0) for y_iter in bootstrap_y]
        #     )
        #     popt_estimates = np.apply_along_axis(
        #         lambda y_iter: fit_exp1B_func(x, y_iter, bounds=[0, 1])[0],
        #         axis=0,
        #         arr=np.array(y_estimates),
        #     )

        # Fit the initial data and compute error bars
        # Where they usng the mean to get a median ???
        # y = [np.mean(y_row) for y_row in y_scatter]

        # If bootstrap was not performed, y_estimates can be inhomogeneous

        # are these the samples ?
        y_estimates = []
        for depth in x:
            y_estimates.append(data.data[qubit, depth])

        # Give the qubit
        error_bars = data_uncertainties(
            y_scatter,
            uncertainties,
            # data_median=y,
            homogeneous=(homogeneous or n_bootstrap != 0),
        )

        import pdb

        pdb.set_trace()

        # Generate symmetric non-zero uncertainty of y for the fit
        # are they overeestimating uncertanties for the fit ???
        sigma = None
        if error_bars is not None:
            sigma = (
                np.max(error_bars, axis=0)
                if isinstance(error_bars[0], Iterable)
                else error_bars
            ) + 0.1

        popt, perr = fit_exp1B_func(x, y_scatter, sigma=sigma, bounds=[0, 1])

        popts[qubit] = popt
        perrs[qubit] = perr

        # Compute fit uncertainties
        if len(popt_estimates):
            perr = data_uncertainties(popt_estimates, uncertainties, data_median=popt)
            perr = perr.T if perr is not None else (0,) * len(popt)

        # Compute the fidelities
        infidelity = (1 - popt[1]) / 2
        fidelity[qubit] = 1 - infidelity
        pulse_fidelity[qubit] = 1 - infidelity / NPULSES_PER_CLIFFORD

        # conversion from np.array to list/tuple
        error_bars = error_bars.tolist() if error_bars is not None else error_bars
        error_barss[qubit] = error_bars

        import pdb

        pdb.set_trace()

        perr = perr if isinstance(perr, tuple) else perr.tolist()

        # Store in dicts the values

    return StandardRBResult(fidelity, pulse_fidelity, popts, perrs, error_barss)


def _plot(data: RBData, fit: StandardRBResult, qubit) -> tuple[list[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        fit (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        tuple[list[go.Figure], str]:
    """

    fig = go.Figure()
    fitting_report = ""

    # Find a better way of recover signals for qubit and depths
    x = data.params["depths"]
    y = data.samples_to_p0s(qubit, x)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    # Create a dictionary for the error bars
    error_y_dict = None
    if fit is not None:
        popt, perr = fit.fit_parameters[qubit], fit.fit_uncertainties[qubit]
        label = "Fit: y=Ap^x<br>A: {}<br>p: {}<br>B: {}".format(
            number_to_str(popt[0], perr[0]),
            number_to_str(popt[1], perr[1]),
            number_to_str(popt[2], perr[2]),
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
        if fit.error_bars is not None:
            error_bars = fit.error_bars[qubit]
            # Constant error bars
            if isinstance(error_bars, Iterable) is False:
                error_y_dict = {"type": "constant", "value": error_bars}
            # Symmetric error bars
            elif isinstance(error_bars[0], Iterable) is False:
                error_y_dict = {"type": "data", "array": error_bars}
            # Asymmetric error bars
            else:
                error_y_dict = {
                    "type": "data",
                    "symmetric": False,
                    "array": error_bars[1],
                    "arrayminus": error_bars[0],
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

    fitting_report = table_html(
        table_dict(
            qubit,
            ["niter", "nshots", "uncertainties", "fidelity", "pulse_fidelity"],
            [
                data.params["niter"],
                data.params["nshots"],
                data.params["uncertainties"],
                number_to_str(
                    fit.fidelity[qubit], np.array(fit.fit_uncertainties[qubit][1]) / 2
                ),
                number_to_str(
                    fit.pulse_fidelity[qubit],
                    np.array(fit.fit_uncertainties[qubit][1])
                    / (2 * NPULSES_PER_CLIFFORD),
                ),
            ],
        )
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",
        xaxis_title="Circuit depth",
        yaxis_title="Survival Probability",
    )

    return [fig], fitting_report


# Build the routine object which is used by qq-auto.
standard_rb = Routine(_acquisition, _fit, _plot)
