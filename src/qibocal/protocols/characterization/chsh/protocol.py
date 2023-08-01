"""Protocol for CHSH experiment using both circuits and pulses."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .circuits import create_chsh_circuits
from .pulses import create_chsh_sequences
from .utils import calculate_frequencies, compute_chsh

COMPUTATIONAL_BASIS = ["00", "01", "10", "11"]


@dataclass
class CHSHParameters(Parameters):
    """Flipping runcard inputs."""

    bell_states: list
    """Bell states to compute CHSH. Using all states is equivalent
        to passing the list [0,1,2,3]."""
    nshots: int
    """Number of shots."""
    ntheta: int
    """Number of angles probed linearly between 0 and 2 pi."""
    native: Optional[bool] = False
    """If True a circuit will be created using only GPI2 and CZ gates."""
    readout_error_model: Optional[str] = None
    """Error mitigation model"""


@dataclass
class CHSHData(Data):
    bell_states: list
    thetas: list
    data: dict[QubitId, QubitId, int, tuple, str] = field(default_factory=dict)
    """Raw data acquired."""

    def register_basis(self, pair, bell_state, basis, frequencies):
        """Store output for single qubit."""

        # Add zero is state do not appear in state
        # could be removed by using high number of shots
        for i in COMPUTATIONAL_BASIS:
            if i not in frequencies:
                frequencies[i] = 0

        # TODO: improve this
        for state, freq in frequencies.items():
            if (pair[0], pair[1], bell_state) not in self.data:
                self.data[pair[0], pair[1], bell_state] = {}
            if basis not in self.data[pair[0], pair[1], bell_state]:
                self.data[pair[0], pair[1], bell_state][basis] = {}
            if state in self.data[pair[0], pair[1], bell_state][basis]:
                self.data[pair[0], pair[1], bell_state][basis][state].append(freq)
            else:
                self.data[pair[0], pair[1], bell_state][basis][state] = [freq]

    def compute_frequencies(self, pair, bell_state):
        freqs = []
        data = self.data[pair[0], pair[1], bell_state]
        for freq_basis in data.values():
            freqs.append(freq_basis)
        return freqs


@dataclass
class CHSHResults(Results):
    entropy: dict[tuple[QubitId, QubitId, int], float] = field(default_factory=dict)


def _acquisition_pulses(
    params: CHSHParameters,
    platform: Platform,
    qubits: Qubits,
) -> CHSHData:
    r""" """

    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = CHSHData(bell_states=params.bell_states, thetas=thetas.tolist())
    for pair in qubits:
        for bell_state in params.bell_states:
            for theta in thetas:
                chsh_sequences = create_chsh_sequences(
                    platform=platform,
                    qubits=pair,
                    theta=theta,
                    bell_state=bell_state,
                )
                for basis, sequence in chsh_sequences.items():
                    results = platform.execute_pulse_sequence(
                        sequence, ExecutionParameters(nshots=params.nshots)
                    )
                    frequencies = calculate_frequencies(
                        results[pair[0]], results[pair[1]]
                    )
                    data.register_basis(pair, bell_state, basis, frequencies)
    return data


def _acquisition_circuits(
    params: CHSHParameters,
    platform: Platform,
    qubits: Qubits,
) -> CHSHData:
    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = CHSHData(bell_states=params.bell_states, thetas=thetas.tolist())
    for pair in qubits:
        for bell_state in params.bell_states:
            for theta in thetas:
                print(pair)
                chsh_circuits = create_chsh_circuits(
                    platform,
                    qubits=pair,
                    bell_state=bell_state,
                    theta=theta,
                    native=params.native,
                    rerr=params.readout_error_model,
                )
                for basis, circuit in chsh_circuits.items():
                    result = circuit(nshots=params.nshots)
                    frequencies = result.frequencies()
                    data.register_basis(pair, bell_state, basis, frequencies)
    return data


def _plot(data: CHSHData, fit: CHSHResults, qubits):
    figures = []

    for bell_state in data.bell_states:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.thetas,
                y=fit.entropy[qubits[0], qubits[1], bell_state],
            )
        )
        figures.append(fig)
    fitting_report = ""

    return figures, fitting_report


def _fit(data: CHSHData) -> CHSHResults:
    results = {}
    for pair in data.pairs:
        for bell_state in data.bell_states:
            freq = data.compute_frequencies(pair, bell_state)
            results[pair[0], pair[1], bell_state] = [
                compute_chsh(freq, bell_state, i) for i in range(len(data.thetas))
            ]
    return CHSHResults(entropy=results)


chsh_circuits = Routine(_acquisition_circuits, _fit, _plot)
"""CHSH experiment using circuits."""
chsh_pulses = Routine(_acquisition_pulses, _fit, _plot)
"""CHSH experiment using pulses."""
