from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Results, Routine
from qibocal.protocols.two_qubit_interaction.utils import order_pair
from qibocal.protocols.utils import HZ_TO_GHZ, table_dict, table_html

from .qubit_flux_dependence import QubitFluxParameters, QubitFluxType
from .qubit_flux_dependence import _acquisition as flux_acquisition

STEP = 60
POINT_SIZE = 10


@dataclass
class AvoidedCrossingParameters(QubitFluxParameters):
    """Avoided Crossing Parameters"""


@dataclass
class AvoidedCrossingResults(Results):
    """Avoided crossing outputs"""

    parabolas: dict[tuple, list]
    """Extracted parabolas"""
    fits: dict[tuple, list]
    """Fits parameters"""
    cz: dict[tuple, list]
    """CZ intersection points """
    iswap: dict[tuple, list]
    """iSwap intersection points"""


@dataclass
class AvoidedCrossingData(Data):
    """Avoided crossing acquisition outputs"""

    qubit_pairs: list
    """list of qubit pairs ordered following the drive frequency"""
    drive_frequency_low: dict = field(default_factory=dict)
    """Lowest drive frequency in each qubit pair"""
    data: dict[tuple[QubitId, str], npt.NDArray[QubitFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: AvoidedCrossingParameters,
    platform: Platform,
    targets: list[QubitPairId],  # qubit pairs
) -> AvoidedCrossingData:
    """
    Data acquisition for avoided crossing.
    This routine performs the qubit flux dependency for the "01" and "02" transition
    on the qubit pair. It returns the bias and frequency values to perform a CZ
    and a iSwap gate.

    Args:
        params (AvoidedCrossingParameters): experiment's parameters.
        platform (Platform): Qibolab platform object.
        qubits (dict): list of targets qubit pairs to perform the action.
    """
    order_pairs = np.array([order_pair(pair, platform) for pair in targets])
    data = AvoidedCrossingData(qubit_pairs=order_pairs.tolist())
    # Extract the qubits in the qubits pairs and evaluate their flux dep
    unique_qubits = np.unique(
        order_pairs[:, 1]
    )  # select qubits with high freq in each couple
    new_qubits = {key: platform.qubits[key] for key in unique_qubits}
    excitations = [Excitations.ge, Excitations.gf]
    for transition in excitations:
        params.transition = transition
        data_transition = flux_acquisition(
            params=params,
            platform=platform,
            targets=new_qubits,
        )
        for qubit in unique_qubits:
            qubit_data = data_transition.data[qubit]
            freq = qubit_data["freq"]
            bias = qubit_data["bias"]
            signal = qubit_data["signal"]
            phase = qubit_data["phase"]
            data.register_qubit(
                QubitFluxType,
                (float(qubit), transition),
                dict(
                    freq=freq.tolist(),
                    bias=bias.tolist(),
                    signal=signal.tolist(),
                    phase=phase.tolist(),
                ),
            )

    unique_low_qubits = np.unique(order_pairs[:, 0])
    data.drive_frequency_low = {
        str(qubit): float(platform.qubits[qubit].drive_frequency)
        for qubit in unique_low_qubits
    }
    return data


def _fit(data: AvoidedCrossingData) -> AvoidedCrossingResults:
    """
    Post-Processing for avoided crossing.
    """
    qubit_data = data.data
    fits = {}
    cz = {}
    iswap = {}
    curves = {tuple(key): find_parabola(val) for key, val in qubit_data.items()}
    for qubit_pair in data.qubit_pairs:
        qubit_pair = tuple(qubit_pair)
        fits[qubit_pair] = {}
        low = qubit_pair[0]
        high = qubit_pair[1]
        # Fit the 02*2 curve
        curve_02 = np.array(curves[high, Excitations.gf]) * 2
        x_02 = np.unique(qubit_data[high, Excitations.gf]["bias"])
        fit_02 = np.polyfit(x_02, curve_02, 2)
        fits[qubit_pair][Excitations.gf] = fit_02.tolist()

        # Fit the 01+10 curve
        curve_01 = np.array(curves[high, Excitations.ge])
        x_01 = np.unique(qubit_data[high, Excitations.ge]["bias"])
        fit_01_10 = np.polyfit(x_01, curve_01 + data.drive_frequency_low[str(low)], 2)
        fits[qubit_pair][Excitations.all_ge] = fit_01_10.tolist()
        # find the intersection of the two parabolas
        delta_fit = fit_02 - fit_01_10
        x1, x2 = solve_eq(delta_fit)
        cz[qubit_pair] = [
            [x1, np.polyval(fit_02, x1)],
            [x2, np.polyval(fit_02, x2)],
        ]
        # find the intersection of the 01 parabola and the 10 line
        fit_01 = np.polyfit(x_01, curve_01, 2)
        fits[qubit_pair][Excitations.ge] = fit_01.tolist()
        fit_pars = deepcopy(fit_01)
        line_val = data.drive_frequency_low[str(low)]
        fit_pars[2] -= line_val
        x1, x2 = solve_eq(fit_pars)
        iswap[qubit_pair] = [[x1, line_val], [x2, line_val]]

    return AvoidedCrossingResults(curves, fits, cz, iswap)


def _plot(
    data: AvoidedCrossingData,
    fit: Optional[AvoidedCrossingResults],
    target: QubitPairId,
):
    """Plotting function for avoided crossing"""
    fitting_report = ""
    figures = []
    order_pair = tuple(index(data.qubit_pairs, target))
    heatmaps = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{i} transition qubit {target[0]}"
            for i in [Excitations.ge, Excitations.gf]
        ],
    )
    parabolas = make_subplots(rows=1, cols=1, subplot_titles=["Parabolas"])
    for i, transition in enumerate([Excitations.ge, Excitations.gf]):
        data_high = data.data[order_pair[1], transition]
        bias_unique = np.unique(data_high.bias)
        min_bias = min(bias_unique)
        max_bias = max(bias_unique)
        plot_heatmap(
            heatmaps, fit, transition, bias_unique, order_pair, data_high, i + 1
        )

    figures.append(heatmaps)

    if fit is not None:
        cz = np.array(fit.cz[order_pair])
        iswap = np.array(fit.iswap[order_pair])
        min_bias = min(min_bias, *cz[:, 0], *iswap[:, 0])
        max_bias = max(max_bias, *cz[:, 0], *iswap[:, 0])
        bias_range = np.linspace(min_bias, max_bias, STEP)
        plot_curves(parabolas, fit, data, order_pair, bias_range)
        plot_intersections(parabolas, cz, iswap)

        parabolas.update_layout(
            xaxis_title="Bias[V]",
            yaxis_title="Frequency[GHz]",
        )
        heatmaps.update_layout(
            coloraxis_colorbar=dict(
                yanchor="top",
                y=1,
                x=-0.08,
                ticks="outside",
            ),
            xaxis_title="Frequency[GHz]",
            yaxis_title="Bias[V]",
            xaxis2_title="Frequency[GHz]",
            yaxis2_title="Bias[V]",
        )
        figures.append(parabolas)
        fitting_report = table_html(
            table_dict(
                target,
                ["CZ bias", "iSwap bias"],
                [np.round(cz[:, 0], 3), np.round(iswap[:, 0], 3)],
            )
        )
    return figures, fitting_report


avoided_crossing = Routine(_acquisition, _fit, _plot)


def find_parabola(data: dict) -> list:
    """
    Finds the parabola in `data`
    """
    freqs = data["freq"]
    currs = data["bias"]
    biass = sorted(np.unique(currs))
    frequencies = []
    for bias in biass:
        data_bias = data[currs == bias]
        index = data_bias["signal"].argmax()
        frequencies.append(freqs[index])
    return frequencies


def solve_eq(pars: list) -> tuple:
    """
    Solver of the quadratic equation

    .. math::
        a x^2 + b x + c = 0

    `pars` is the list [a, b, c].
    """
    first_term = -1 * pars[1]
    second_term = np.sqrt(pars[1] ** 2 - 4 * pars[0] * pars[2])
    x1 = (first_term + second_term) / pars[0] / 2
    x2 = (first_term - second_term) / pars[0] / 2
    return x1, x2


def index(pairs: list, item: list) -> list:
    """Find the ordered pair"""
    for pair in pairs:
        if set(pair) == set(item):
            return pair
    raise ValueError(f"{item} not in pairs")


class Excitations(str, Enum):
    """
    Excited two qubits states.
    """

    ge = "01"
    """First qubit in ground state, second qubit in excited state"""
    gf = "02"
    """First qubit in ground state, second qubit in the first excited state out
    of the computational basis."""
    all_ge = "01+10"
    """One of the qubit in the ground state and the other one in the excited state."""


def plot_heatmap(heatmaps, fit, transition, bias_unique, order_pair, data_high, col):
    heatmaps.add_trace(
        go.Heatmap(
            x=data_high.freq * HZ_TO_GHZ,
            y=data_high.bias,
            z=data_high.signal,
            coloraxis="coloraxis",
        ),
        row=1,
        col=col,
    )
    if fit is not None:
        # the fit of the parabola in 02 transition was done doubling the frequencies
        heatmaps.add_trace(
            go.Scatter(
                x=np.polyval(fit.fits[order_pair][transition], bias_unique)
                / col
                * HZ_TO_GHZ,
                y=bias_unique,
                mode="markers",
                marker_color="lime",
                showlegend=True,
                marker=dict(size=POINT_SIZE),
                name=f"Curve estimation {transition}",
            ),
            row=1,
            col=col,
        )
        heatmaps.add_trace(
            go.Scatter(
                x=np.array(fit.parabolas[order_pair[1], transition]) * HZ_TO_GHZ,
                y=bias_unique,
                mode="markers",
                marker_color="black",
                showlegend=True,
                marker=dict(symbol="cross", size=POINT_SIZE),
                name=f"Parabola {transition}",
            ),
            row=1,
            col=col,
        )


def plot_curves(parabolas, fit, data, order_pair, bias_range):
    for transition in [Excitations.ge, Excitations.gf, Excitations.all_ge]:
        parabolas.add_trace(
            go.Scatter(
                x=bias_range,
                y=np.polyval(fit.fits[order_pair][transition], bias_range) * HZ_TO_GHZ,
                showlegend=True,
                name=transition,
            )
        )
    parabolas.add_trace(
        go.Scatter(
            x=bias_range,
            y=np.array([data.drive_frequency_low[str(order_pair[0])]] * STEP)
            * HZ_TO_GHZ,
            showlegend=True,
            name="10",
        )
    )


def plot_intersections(parabolas, cz, iswap):
    parabolas.add_trace(
        go.Scatter(
            x=cz[:, 0],
            y=cz[:, 1] * HZ_TO_GHZ,
            showlegend=True,
            name="CZ",
            marker_color="black",
            mode="markers",
            marker=dict(symbol="cross", size=POINT_SIZE),
        )
    )
    parabolas.add_trace(
        go.Scatter(
            x=iswap[:, 0],
            y=iswap[:, 1] * HZ_TO_GHZ,
            showlegend=True,
            name="iswap",
            marker_color="blue",
            mode="markers",
            marker=dict(symbol="cross", size=10),
        )
    )
