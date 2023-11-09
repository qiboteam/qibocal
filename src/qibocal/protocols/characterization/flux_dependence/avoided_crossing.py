from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, QubitsPairs, Results, Routine
from qibocal.protocols.characterization.two_qubit_interaction.utils import order_pair

from .qubit_flux_dependence import QubitFluxParameters, QubitFluxType
from .qubit_flux_dependence import _acquisition as flux_acquisition

STEP = 60
POINT_SIZE = 10


@dataclass
class AvoidCrossParameters(QubitFluxParameters):
    ...


@dataclass
class AvoidCrossResults(Results):
    parabolas: dict
    # TODO: doc
    fits: dict
    cz: dict
    iswap: dict


@dataclass
class AvoidCrossData(Data):
    qubit_pairs: list
    ro_freq_high: dict = field(default_factory=dict)
    data: dict[tuple[QubitId, str], npt.NDArray[QubitFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: AvoidCrossParameters,
    platform: Platform,
    qubits: QubitsPairs,  # qubit pairs
) -> AvoidCrossData:
    qubit_pairs = list(qubits.keys())
    order_pairs = np.array([order_pair(pair, platform.qubits) for pair in qubit_pairs])
    data = AvoidCrossData(qubit_pairs=order_pairs.tolist())
    # Extract the qubits in the qubits pairs and evaluate their flux dep
    # qubits_keys = list(qubits.keys())
    unique_qubits = np.unique(
        order_pairs[:, 0]
    )  # select qubits with lower freq in each couple
    new_qubits = {key: platform.qubits[key] for key in unique_qubits}

    for transition in ["01", "02"]:
        params.transition = transition
        data_transition = flux_acquisition(
            params=params,
            platform=platform,
            qubits=new_qubits,
        )
        for qubit in unique_qubits:
            qubit_data = data_transition.data[qubit]
            freq = qubit_data["freq"]
            bias = qubit_data["bias"]
            msr = qubit_data["msr"]
            phase = qubit_data["phase"]
            data.register_qubit(
                QubitFluxType,
                (float(qubit), transition),
                dict(
                    freq=freq.tolist(),
                    bias=bias.tolist(),
                    msr=msr.tolist(),
                    phase=phase.tolist(),
                ),
            )

    unique_high_qubits = np.unique(order_pairs[:, 1])
    data.ro_freq_high = {
        float(qubit): float(platform.qubits[qubit].readout_frequency)
        for qubit in unique_high_qubits
    }
    return data


def _fit(data: AvoidCrossData) -> AvoidCrossResults:
    # qubits = data.qubits
    qubit_data = data.data
    fits = {}
    cz = {}
    iswap = {}
    curves = {key: find_parabola(val) for key, val in qubit_data.items()}
    for qubit_pair in data.qubit_pairs:
        qubit_pair = tuple(qubit_pair)
        low = qubit_pair[0]
        high = qubit_pair[1]
        # Fit the 02*2 curve
        curve_02 = np.array(curves[low, "02"]) * 2
        x_02 = np.unique(qubit_data[low, "02"]["bias"])
        fit_02 = np.polyfit(x_02, curve_02, 2)
        fits[qubit_pair, "02"] = fit_02.tolist()

        # Fit the 01+10 curve
        curve_01 = np.array(curves[low, "01"])
        x_01 = np.unique(qubit_data[low, "01"]["bias"])
        fit_01_10 = np.polyfit(x_01, curve_01 + data.ro_freq_high[high], 2)
        fits[qubit_pair, "01+10"] = fit_01_10.tolist()
        # find the intersection of the two parabolas
        delta_fit = fit_02 - fit_01_10
        x1, x2 = solve_eq(delta_fit)
        cz[qubit_pair] = [
            [x1, np.polyval(fit_02, x1)],
            [x2, np.polyval(fit_02, x2)],
        ]
        # find the intersection of the 01 parabola and the 10 line
        fit_01 = np.polyfit(x_01, curve_01, 2)
        fits[qubit_pair, "01"] = fit_01.tolist()
        fit_pars = deepcopy(fit_01)
        line_val = data.ro_freq_high[high]
        fit_pars[2] -= line_val
        x1, x2 = solve_eq(fit_pars)
        iswap[qubit_pair] = [[x1, line_val], [x2, line_val]]
        import matplotlib.pyplot as plt

        plt.plot(x_02, curve_02)
        plt.scatter(x_02, np.polyval(fit_02, x_02))
        plt.title(f"{qubit_pair}")
        plt.savefig(f"{qubit_pair}")

    return AvoidCrossResults(curves, fits, cz, iswap)


def _plot(data: AvoidCrossData, fit: AvoidCrossResults, qubit):
    fitting_report = ""
    figures = []
    order_pair = tuple(index(data.qubit_pairs, qubit))
    heatmaps = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{i} transition qubit {qubit[0]}" for i in ["01", "02"]],
    )
    parabolas = make_subplots(rows=1, cols=1, subplot_titles=["Parabolas"])
    for i, transition in enumerate(["01", "02"]):
        data_low = data.data[order_pair[0], transition]
        bias_unique = np.unique(data_low.bias)
        min_bias = min(bias_unique)
        max_bias = max(bias_unique)
        heatmaps.add_trace(
            go.Heatmap(
                x=data_low.freq,
                y=data_low.bias,
                z=data_low.msr,
                coloraxis="coloraxis",
            ),
            row=1,
            col=i + 1,
        )

        # the fit of the parabola in 02 transition was done doubling the frequencies
        heatmaps.add_trace(
            go.Scatter(
                x=np.polyval(fit.fits[order_pair, transition], data_low.bias) / (i + 1),
                y=bias_unique,
                mode="markers",
                marker_color="lime",
                showlegend=True,
                marker=dict(size=POINT_SIZE),
                name=f"Curve estimation {transition}",
            ),
            row=1,
            col=i + 1,
        )
        heatmaps.add_trace(
            go.Scatter(
                x=fit.parabolas[order_pair[0], transition],
                y=bias_unique,
                mode="markers",
                marker_color="turquoise",
                showlegend=True,
                marker=dict(symbol="cross", size=POINT_SIZE),
                name=f"Parabola {transition}",
            ),
            row=1,
            col=i + 1,
        )
    cz = np.array(fit.cz[order_pair])
    iswap = np.array(fit.iswap[order_pair])
    min_bias = min(min_bias, *cz[:, 0], *iswap[:, 0])
    max_bias = max(max_bias, *cz[:, 0], *iswap[:, 0])
    bias_range = np.linspace(min_bias, max_bias, STEP)
    for transition in ["01", "02", "01+10"]:
        parabolas.add_trace(
            go.Scatter(
                x=bias_range,
                y=np.polyval(fit.fits[order_pair, transition], bias_range),
                showlegend=True,
                name=transition,
            )
        )
    parabolas.add_trace(
        go.Scatter(
            x=bias_range,
            y=[data.ro_freq_high[order_pair[1]]] * STEP,
            showlegend=True,
            name="10",
        )
    )
    parabolas.add_trace(
        go.Scatter(
            x=cz[:, 0],
            y=cz[:, 1],
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
            y=iswap[:, 1],
            showlegend=True,
            name="iswap",
            marker_color="blue",
            mode="markers",
            marker=dict(symbol="cross", size=10),
        )
    )
    heatmaps.update_layout(
        coloraxis_colorbar=dict(
            yanchor="top",
            y=1,
            x=-0.08,
            ticks="outside",
        )
    )
    figures.append(heatmaps)
    figures.append(parabolas)
    return figures, fitting_report


avoided_crossing = Routine(_acquisition, _fit, _plot)


def find_parabola(data):
    freqs = data["freq"]
    currs = data["bias"]
    # filter1 = (freqs < max_freq) & (currs > min_bias) & (currs < max_bias)
    # filtered = data[filter1]
    biass = sorted(np.unique(currs))
    frequencies = []
    for bias in biass:
        index = data[currs == bias]["msr"].argmax()
        frequencies.append(freqs[index])
    return frequencies


def solve_eq(pars):
    first_term = -1 * pars[1]
    second_term = np.sqrt(pars[1] ** 2 - 4 * pars[0] * pars[2])
    x1 = (first_term + second_term) / pars[0] / 2
    x2 = (first_term - second_term) / pars[0] / 2
    return x1, x2


def index(pairs, item):
    for pair in pairs:
        if set(pair) == set(item):
            return pair
