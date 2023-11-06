from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, QubitsPairs, Results, Routine
from qibocal.protocols.characterization.two_qubit_interaction.utils import order_pair

from .qubit_flux_dependence import QubitFluxData, QubitFluxParameters, QubitFluxType
from .qubit_flux_dependence import _acquisition as flux_acquisition


@dataclass
class AvoidCrossParameters(QubitFluxParameters):
    ...


@dataclass
class AvoidCrossResults(Results):
    parabolas: dict[str, dict]
    # TODO: doc
    fits: dict


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
    print("PPPPPPPP", qubit_pairs, order_pairs)
    data = AvoidCrossData(qubit_pairs=order_pairs)
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
            print(freq.dtype, bias.dtype, msr.dtype, phase.dtype)
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
    return data


def _fit(data: QubitFluxData) -> AvoidCrossResults:
    # qubits = data.qubits
    qubit_data = data.data
    fits = {}
    for couple in data.qubit_pairs:
        curves = {key: find_parabola(val) for key, val in qubit_data.items()}
        print(curves)
        for state in ["01", "02"]:
            for qubit in data.qubits:
                x = np.unique(
                    qubit_data[qubit, state]["bias"]
                )  # - np.mean(data[state]["bias"])
                y = np.copy(curves[qubit, state])
                print(x, y, qubit_data[qubit, state])
                fits[qubit, state] = np.polyfit(x, y, 2)
                pred = sum(p * x ** (2 - i) for i, p in enumerate(fits[qubit, state]))

    return AvoidCrossResults(curves, fits)


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
