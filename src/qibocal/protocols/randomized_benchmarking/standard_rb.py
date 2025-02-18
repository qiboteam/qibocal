from dataclasses import dataclass
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Parameters, QubitId, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import table_dict, table_html
from .fitting import exp1B_func, fit_exp1B_func
from .utils import RBData, data_uncertainties, number_to_str, rb_acquisition

NPULSES_PER_CLIFFORD = 23 * 2 / 24  # 1.875


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
    uncertainties: Optional[float] = None
    """Method of computing the error bars of the signal and uncertainties of the fit. If ``None``,
    it computes the standard deviation. Otherwise it computes the corresponding confidence interval. Defaults `None`."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple circuits in a single instrument call.
    Defaults to ``False``."""
    seed: Optional[int] = None
    """A fixed seed to initialize ``np.random.Generator``.

    If ``None``, uses a random seed.
    Defaults is ``None``.
    """
    nshots: int = 10
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

    # FIXME: fix this after https://github.com/qiboteam/qibocal/pull/597
    def __contains__(self, qubit: QubitId):
        return True


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
        - gate_list (list): List of gates.

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
) -> Iterable:
    """Returns single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        targets (list[QubitId]):
            list of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    circuits = []
    indexes = defaultdict(list)
    for _ in range(niter):
        for target in targets:
            circuit, random_index = layer_circuit(rb_gen, depth, target)
            add_inverse_layer(circuit)
            add_measurement_layer(circuit)
            if noise_model is not None:
                circuit = noise_model.apply(circuit)
            circuits.append(circuit)
            indexes[target].append(random_index)

    return circuits, indexes


def _acquisition(
    params: StandardRBParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params: All parameters in one object.
        platform: CalibrationPlatform the experiment is executed on.
        target: list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    return rb_acquisition(params, targets)


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    qubits = data.qubits

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
        pulse_fidelity[qubit] = 1 - infidelity / NPULSES_PER_CLIFFORD

        # conversion from np.array to list/tuple
        error_bars = error_bars.tolist()
        error_barss[qubit] = error_bars
        perrs[qubit] = perr
        popts[qubit] = popt

    return StandardRBResult(fidelity, pulse_fidelity, popts, perrs, error_barss)


def _plot(
    data: RBData, fit: StandardRBResult, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        fit (StandardRBResult): Is called for the plot.
        target (_type_): Not used yet.

    Returns:
        tuple[list[go.Figure], str]:
    """
    target = tuple(target) if isinstance(target, list) else target
    qubit = target
    fig = go.Figure()
    fitting_report = ""
    x = data.depths
    raw_data = data.extract_probabilities(qubit)
    y = np.mean(raw_data, axis=1)
    raw_depths = [[depth] * data.niter for depth in data.depths]

    fig.add_trace(
        go.Scatter(
            x=np.hstack(raw_depths),
            y=np.hstack(raw_data),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="iterations",
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
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                qubit,
                ["niter", "nshots", "uncertainties", "fidelity", "pulse_fidelity"],
                [
                    data.niter,
                    data.nshots,
                    data.uncertainties,
                    number_to_str(
                        fit.fidelity[qubit],
                        np.array(fit.fit_uncertainties[qubit][1]) / 2,
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
        xaxis_title="Circuit depth",
        yaxis_title="Survival Probability",
    )

    return [fig], fitting_report


def _update(results: StandardRBResult, platform: CalibrationPlatform, target: QubitId):
    """Write rb fidelity in calibration."""

    # TODO: shall we use the gate fidelity or the pulse fidelity
    platform.calibration.single_qubits[target].rb_fidelity = (
        results.fidelity[target],
        results.fit_uncertainties[target][1] / 2,
    )


standard_rb = Routine(_acquisition, _fit, _plot, _update)
