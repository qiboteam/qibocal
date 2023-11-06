from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, QubitsPairs, Results, Routine
from qibocal.protocols.characterization.two_qubit_interaction.utils import order_pair

from .qubit_flux_dependence import QubitFluxParameters, QubitFluxType
from .qubit_flux_dependence import _acquisition as flux_acquisition


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
    for qubit in unique_high_qubits:
        data.register_qubit(
            QubitFluxType,
            (float(qubit), "01"),
            dict(
                freq=[platform.qubits[qubit].readout_frequency],
                bias=[0],
                msr=[0],
                phase=[0],
            ),
        )
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
        fit_01_10 = np.polyfit(x_01, curve_01 + qubit_data[high, "01"]["freq"][0], 2)
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
        line_val = qubit_data[high, "01"]["freq"][0]
        fit_pars[2] -= line_val
        x1, x2 = solve_eq(fit_pars)
        iswap[qubit_pair] = [[x1, line_val], [x2, line_val]]
    return AvoidCrossResults(curves, fits, cz, iswap)


def _plot(data: AvoidCrossData, fit: AvoidCrossResults, qubit):
    pass


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
