from dataclasses import dataclass, field
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Routine

from ..utils import table_dict, table_html
from .fitting import exp1B_func
from .utils import RBData, StandardRBResult, fit, number_to_str, rb_acquisition


class Depthsdict(TypedDict):
    """Dictionary used to build a list of depths as ``range(start, stop,
    step)``."""

    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    depths: Union[list, Depthsdict]
    """A list of depths/sequence lengths.

    If a dictionary is given the list will be build.
    """
    niter: int
    """Sets how many iterations over the same depth value."""
    uncertainties: Optional[float] = None
    """Method of computing the error bars of the signal and uncertainties of
    the fit.

    If ``None``,
    it computes the standard deviation. Otherwise it computes the corresponding confidence interval. Defaults `None`.
    """
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple circuits in a
    single instrument call.

    Defaults to ``False``.
    """
    seed: Optional[int] = None
    """A fixed seed to initialize ``np.random.Generator``.

    If ``None``, uses a random seed.
    Defaults is ``None``.
    """
    noise_model: Optional[str] = None
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.randomized_benchmarking.noisemodels`"""
    noise_params: Optional[list] = field(default_factory=list)
    """With this the noise model will be initialized, if not given random
    values will be used."""
    nshots: int = 10
    """Just to add the default value."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params: All parameters in one object.
        platform: Platform the experiment is executed on.
        target: list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    return rb_acquisition(params, platform, targets)


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with
    an exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    return fit(data.qubits, data)


def _plot(
    data: RBData, fit: StandardRBResult, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result
    object and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        fit (StandardRBResult): Is called for the plot.
        target (_type_): Not used yet.

    Returns:
        tuple[list[go.Figure], str]:
    """
    if isinstance(target, list):
        target = tuple(target)
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
                str(qubit),
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
                        / (2 * data.npulses_per_clifford),
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


standard_rb = Routine(_acquisition, _fit, _plot)
