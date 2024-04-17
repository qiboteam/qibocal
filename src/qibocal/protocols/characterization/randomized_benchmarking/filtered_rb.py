from dataclasses import dataclass

import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Routine
from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    rb_acquisition,
)

from .standard_rb import FilteredRBResult, RBData, StandardRBResult


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
        target (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    return rb_acquisition(params, targets, add_inverse_layer=False)


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
