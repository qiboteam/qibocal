"""Protocol for CHSH experiment using both circuits and pulses."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibo.backends import construct_backend
from qibolab import Platform

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit

from .circuits import create_chsh_circuits
from .utils import READOUT_BASIS, compute_chsh

COMPUTATIONAL_BASIS = ["00", "01", "10", "11"]

CLASSICAL_BOUND = 2
"""Classical limit of CHSH,"""
QUANTUM_BOUND = 2 * np.sqrt(2)
"""Quantum limit of CHSH."""

DataType = dict[QubitId, QubitId, int, tuple, str]
FreqType = dict[int, list[dict[str, list[int]]]]


@dataclass
class CHSHParameters(Parameters):
    """CHSH runcard inputs."""

    bell_states: list[int]
    """List with Bell states to compute CHSH.
    The following notation it is used:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    """
    ntheta: int
    """Number of angles probed linearly between 0 and 2 pi."""
    native: Optional[bool] = True
    """If True a circuit will be created using only GPI2 and CZ gates."""


def merge_frequencies(data: DataType, pair: tuple[QubitId, QubitId], bell_state: int):
    """Merge frequencies with different measurement basis."""
    freqs = []
    bell_data = {
        (index[3], index[4]): value
        for index, value in data.items()
        if index[:3] == (pair[0], pair[1], bell_state)
    }

    freqs = []
    for i in READOUT_BASIS:
        freqs.append(
            {
                state[1]: value.tolist()
                for state, value in bell_data.items()
                if state[0] == i
            }
        )
    return freqs


def mitigated_frequencies(frequencies, mitigation_matrix, thetas):
    mitigated_freq_list = []
    for freq_basis in frequencies:
        mitigated_freq = {format(i, f"0{2}b"): [] for i in range(4)}
        for i in range(len(thetas)):
            freq_array = np.zeros(4)
            for k, v in freq_basis.items():
                freq_array[int(k, 2)] = v[i]
            freq_array = freq_array.reshape(-1, 1)
            for j, val in enumerate(mitigation_matrix @ freq_array):
                mitigated_freq[format(j, f"0{2}b")].append(float(val))
        mitigated_freq_list.append(mitigated_freq)
    return mitigated_freq_list


@dataclass
class CHSHData(Data):
    """CHSH Data structure."""

    bell_states: list[int]
    """Bell states list."""
    thetas: list
    """Angles probed."""
    data: DataType = field(default_factory=dict)
    """Raw data acquired."""
    frequencies: FreqType = field(default_factory=dict)
    mitigated_frequencies: FreqType = field(default_factory=dict)

    def register_basis(self, pair, bell_state, basis, frequencies):
        """Store output for single qubit."""

        # Add zero is state do not appear in state
        # could be removed by using high number of shots
        for i in COMPUTATIONAL_BASIS:
            if i not in frequencies:
                frequencies[i] = 0

        for state, freq in frequencies.items():
            if (pair[0], pair[1], bell_state, basis, state) in self.data:
                self.data[pair[0], pair[1], bell_state, basis, state] = np.concatenate(
                    (
                        self.data[pair[0], pair[1], bell_state, basis, state],
                        np.array([freq]),
                    )
                )
            else:
                self.data[pair[0], pair[1], bell_state, basis, state] = np.array([freq])


@dataclass
class CHSHResults(Results):
    """CHSH Results class."""

    chsh: dict[tuple[QubitPairId, int], float] = field(default_factory=dict)
    """Raw CHSH value."""
    chsh_mitigated: dict[tuple[QubitPairId, int], float] = field(default_factory=dict)
    """Mitigated CHSH value."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.

        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.

        """

        return key in [(target, control) for target, control, _ in self.chsh]


def _acquisition(
    params: CHSHParameters,
    platform: Platform,
    targets: list[list[QubitId]],
) -> CHSHData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""
    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = CHSHData(bell_states=params.bell_states, thetas=thetas.tolist())

    backend = construct_backend("qibolab", platform=platform)
    transpiler = dummy_transpiler(backend)
    for pair in targets:
        try:
            mitigation_matrix = (
                platform.calibration.get_readout_mitigation_matrix_element(pair)
            )
        except AssertionError:
            mitigation_matrix = None

        for bell_state in params.bell_states:
            for theta in thetas:
                chsh_circuits = create_chsh_circuits(
                    bell_state=bell_state,
                    theta=theta,
                    native=params.native,
                )
                for basis, circuit in chsh_circuits.items():
                    _, result = execute_transpiled_circuit(
                        circuit,
                        pair,
                        backend,
                        transpiler=transpiler,
                        nshots=params.nshots,
                    )
                    frequencies = result.frequencies()
                    data.register_basis(pair, bell_state, basis, frequencies)

            data.frequencies[bell_state] = freqs = merge_frequencies(
                data.data, pair, bell_state
            )
            if mitigation_matrix is not None:
                data.mitigated_frequencies[bell_state] = mitigated_frequencies(
                    freqs, mitigation_matrix, thetas
                )

    return data


def _plot(data: CHSHData, fit: CHSHResults, target: QubitPairId):
    """Plotting function for CHSH protocol."""
    figures = []
    for bell_state in data.bell_states:
        fig = go.Figure(layout_yaxis_range=[-3, 3])
        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=data.thetas,
                    y=fit.chsh[target[0], target[1], bell_state],
                    name="Bare",
                )
            )
            if fit.chsh_mitigated:
                fig.add_trace(
                    go.Scatter(
                        x=data.thetas,
                        y=fit.chsh_mitigated[target[0], target[1], bell_state],
                        name="Mitigated",
                    )
                )

        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.thetas,
                y=[+CLASSICAL_BOUND] * len(data.thetas),
                line_color="gray",
                name="Classical limit",
                line_dash="dash",
                legendgroup="classic",
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.thetas,
                y=[-CLASSICAL_BOUND] * len(data.thetas),
                line_color="gray",
                name="Classical limit",
                legendgroup="classic",
                line_dash="dash",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.thetas,
                y=[+QUANTUM_BOUND] * len(data.thetas),
                line_color="gray",
                name="Quantum limit",
                legendgroup="quantum",
            )
        )

        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.thetas,
                y=[-QUANTUM_BOUND] * len(data.thetas),
                line_color="gray",
                name="Quantum limit",
                legendgroup="quantum",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis_title="Theta [rad]",
            yaxis_title="CHSH value",
            xaxis=dict(range=[min(data.thetas), max(data.thetas)]),
        )
        figures.append(fig)

    return figures, ""


def _fit(data: CHSHData) -> CHSHResults:
    """Fitting for CHSH protocol."""
    results = {}
    mitigated_results = {}
    # patch for fixing the plot to appear when qubits are given in non-sorted order
    pairs = list({tuple(q[:2]) for q in data.data})
    for pair in pairs:
        for bell_state in data.bell_states:
            freq = data.frequencies[bell_state]
            results[pair[0], pair[1], bell_state] = [
                compute_chsh(freq, bell_state, ith) for ith in range(len(data.thetas))
            ]

            if bell_state in data.mitigated_frequencies:
                mitigated_freq = data.mitigated_frequencies[bell_state]
                mitigated_results[pair[0], pair[1], bell_state] = [
                    compute_chsh(mitigated_freq, bell_state, ith)
                    for ith in range(len(data.thetas))
                ]
    return CHSHResults(chsh=results, chsh_mitigated=mitigated_results)


chsh = Routine(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CHSH experiment using pulses."""
