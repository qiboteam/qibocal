from dataclasses import dataclass
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    StandardRBResult,
)

from ..utils import table_dict, table_html
from .fitting import exp1B_func
from .utils import RB2QData, fit, number_to_str, twoq_rb_acquisition


@dataclass
class StandardRB2QParameters(StandardRBParameters):
    """Parameters for the standard 2q randomized benchmarking protocol."""

    file: str = "2qubitCliffs.json"
    """File with the cliffords to be used."""
    file_inv: str = "2qubitCliffsInv.npz"
    """File with the cliffords to be used in an inverted dict."""


def _acquisition(
    params: StandardRB2QParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QData:
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

    return twoq_rb_acquisition(params, targets)


def _fit(data: RB2QData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    qubits = data.pairs
    results = fit(qubits, data)

    return results


def _plot(
    data: RB2QData, fit: StandardRBResult, target: QubitPairId
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

    qubits = target
    fig = go.Figure()
    fitting_report = ""
    x = data.depths
    raw_data = data.extract_probabilities(qubits)
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
        popt, perr = fit.fit_parameters[qubits], fit.fit_uncertainties[qubits]
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
            error_bars = fit.error_bars[qubits]
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
                str(qubits),
                ["niter", "nshots", "uncertainties", "fidelity", "pulse_fidelity"],
                [
                    data.niter,
                    data.nshots,
                    data.uncertainties,
                    number_to_str(
                        fit.fidelity[qubits],
                        np.array(fit.fit_uncertainties[qubits][1]) / 2,
                    ),
                    number_to_str(
                        fit.pulse_fidelity[qubits],
                        np.array(fit.fit_uncertainties[qubits][1])
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


standard_rb_2q = Routine(_acquisition, _fit, _plot)
