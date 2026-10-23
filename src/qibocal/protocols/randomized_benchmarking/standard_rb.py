from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Parameters, Protocol, QubitId
from qibocal.calibration import CalibrationPlatform

from ..utils import table_dict, table_html
from .fitting import exp1B_func
from .utils import (
    RBData,
    StandardRBResult,
    fit,
    number_to_str,
    rb_acquisition,
)

__all__ = ["RBData", "StandardRBParameters", "standard_rb"]


class Depthsdict(TypedDict):
    """Dictionary used to build a list of depths as ``range(start, stop,
    step)``."""

    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    depths: list | Depthsdict
    """A list of depths/sequence lengths.

    If a dictionary is given the list will be build.
    """
    niter: int
    """Sets how many iterations over the same depth value."""
    seed: int | None = None
    """A fixed seed to initialize ``np.random.Generator``.

    If ``None``, uses a random seed.
    Defaults is ``None``.
    """
    nshots: int = 10
    """Just to add the default value."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


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

    return rb_acquisition(params, platform, targets)


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with
    an exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    return fit(data)


def survival_figure(
    data: RBData, fit: StandardRBResult | None, target, single_qubit: bool
) -> go.Figure:
    """Builds the survival probability figure, with the fit when available.

    Args:
        data (RBData): Data object with the survival probabilities.
        fit (StandardRBResult, optional): Fit results, ``None`` if not available.
        target: Qubit or qubit pair to plot.
        single_qubit (bool): Whether ``target`` is a single qubit or a qubit pair.

    Returns:
        go.Figure: Survival probability versus circuit depth.
    """
    qubit = target
    fig = go.Figure()
    x = data.depths
    raw_data = np.array(
        [
            val["survival_probs"]
            for key, val in data.data.items()
            if (key[0] if single_qubit else key[:2]) == target
        ]
    )  # rows -> depths, cols -> iterations
    y = np.mean(raw_data, axis=1)
    raw_depths = [[depth] * data.niter for depth in data.depths]

    fig.add_trace(
        go.Scatter(
            x=np.hstack(raw_depths),
            y=np.hstack(raw_data),
            line={"color": "#6597aa"},
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="iterations",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line={"color": "#aa6464"},
            mode="markers",
            name="average",
        )
    )
    # Create a dictionary for the error bars
    error_y_dict = None
    if fit is not None:
        popt, perr = fit.fit_parameters[qubit], fit.fit_uncertainties[qubit]
        label = f"Fit: y = A p^x + B<br>A: {number_to_str(popt[0], perr[0])}<br>p: {number_to_str(popt[1], perr[1])}<br>B: {number_to_str(popt[2], perr[2])}"
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

    fig.update_layout(
        showlegend=True,
        xaxis_title="Circuit depth",
        yaxis_title="Survival Probability",
    )

    return fig


def _plot(
    data: RBData, fit: StandardRBResult | None, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Survival probability figure and fidelity report of a standard RB experiment.

    Args:
        data (RBData): Data object with the survival probabilities.
        fit (StandardRBResult, optional): Fit results, ``None`` if not available.
        target (QubitId): Qubit to plot.

    Returns:
        tuple[list[go.Figure], str]: The survival figure and the HTML report.
    """
    target = tuple(target) if isinstance(target, list) else target
    qubit = target
    fig = survival_figure(data, fit, target, single_qubit=isinstance(target, int))
    metadata = table_html(
        table_dict(str(qubit), ["niter", "nshots"], [data.niter, data.nshots])
    )

    model_note = (
        "<p>(d = 2): F_Cl = 1 - (d - 1) / d * (1 - p) = (1 + p) / 2 <br>"
        "F_avg,gate ≈ 1 - (1 - F_Cl) / N (first order), with N = {npulses} gates per "
        "Clifford</p>"
    ).format(npulses=f"{data.npulses_per_clifford:g}")

    fitting_report = ""
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                str(qubit),
                ["F_Cl", "F_avg,gate"],
                [
                    (fit.fidelity[qubit], fit.fidelity_uncertainty[qubit]),
                    (fit.pulse_fidelity[qubit], fit.pulse_fidelity_uncertainty[qubit]),
                ],
                display_error=True,
            )
        )

    return [fig], metadata + fitting_report + model_note


def _update(results: StandardRBResult, platform: CalibrationPlatform, target: QubitId):
    """Write rb fidelity in calibration."""

    # TODO: shall we use the gate fidelity or the pulse fidelity
    platform.calibration.single_qubits[target].rb_fidelity = (
        results.fidelity[target],
        results.fidelity_uncertainty[target],
    )


standard_rb = Protocol(_acquisition, _fit, _plot, _update)
