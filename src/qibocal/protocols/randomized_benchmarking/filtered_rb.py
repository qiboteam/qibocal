from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Results, Routine
from qibocal.protocols.randomized_benchmarking.utils import rb_acquisition
from qibocal.protocols.utils import table_dict, table_html

from .standard_rb import RBData, StandardRBParameters


@dataclass
class FilteredRBParameters(StandardRBParameters):
    """Filtered Randomized Benchmarking runcard inputs."""


@dataclass
class FilteredRBResult(Results):
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
        params : All parameters in one object.
        platform : Platform the experiment is executed on.
        target : list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    return rb_acquisition(params, platform, targets, add_inverse_layer=False)


def _fit(data: RBData) -> FilteredRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        FilteredRBResult: Aggregated and processed data.
    """
    return FilteredRBResult()


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

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                qubit,
                ["niter", "nshots"],
                [
                    data.niter,
                    data.nshots,
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Circuit depth",
        yaxis_title="Survival Probability",
    )

    return [fig], fitting_report


filtered_rb = Routine(_acquisition, _fit, _plot)
