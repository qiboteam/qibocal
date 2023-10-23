"""Protocol for CHSH experiment using both circuits and pulses."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ...readout_mitigation_matrix import (
    ReadoutMitigationMatrixParameters as mitigation_params,
)
from ...readout_mitigation_matrix import _acquisition as mitigation_acquisition
from ...readout_mitigation_matrix import _fit as mitigation_fit
from ...utils import calculate_frequencies
from .circuits import create_chsh_circuits
from .pulses import create_chsh_sequences
from .utils import READOUT_BASIS, compute_chsh

COMPUTATIONAL_BASIS = ["00", "01", "10", "11"]


@dataclass
class CHSHParameters(Parameters):
    """CHSH runcard inputs."""

    bell_states: list
    """List with Bell states to compute CHSH.
    The following notation it is used:
    0 -> |00>+|11>
    1 -> |00>-|11>
    2 -> |10>-|01>
    3 -> |10>+|01>
    """
    ntheta: int
    """Number of angles probed linearly between 0 and 2 pi."""
    native: Optional[bool] = False
    """If True a circuit will be created using only GPI2 and CZ gates."""
    apply_error_mitigation: Optional[bool] = False
    """Error mitigation model"""


@dataclass
class CHSHData(Data):
    bell_states: list
    """Bell states list."""
    thetas: list
    """Angles probed."""
    data: dict[QubitId, QubitId, int, tuple, str] = field(default_factory=dict)
    """Raw data acquired."""
    mitigation_matrix: dict[tuple[QubitId, ...], npt.NDArray] = field(
        default_factory=dict
    )
    """Mitigation matrix computed using the readout_mitigation_matrix protocol."""

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

    def merge_frequencies(self, pair, bell_state):
        """Merge frequencies with different measurement basis."""
        freqs = []
        bell_data = {
            (index[3], index[4]): value
            for index, value in self.data.items()
            if index[:3] == (pair[0], pair[1], bell_state)
        }

        freqs = []
        for i in READOUT_BASIS:
            freqs.append(
                {state[1]: value for state, value in bell_data.items() if state[0] == i}
            )

        return freqs

    @property
    def global_params(self):
        """Convert non-arrays attributes into dict."""
        data_dict = super().global_params
        data_dict.pop("mitigation_matrix")

        return data_dict


@dataclass
class CHSHResults(Results):
    chsh: dict[tuple[QubitId, QubitId, int], float] = field(default_factory=dict)
    chsh_mitigated: dict[tuple[QubitId, QubitId, int], float] = field(
        default_factory=dict
    )


def _acquisition_pulses(
    params: CHSHParameters,
    platform: Platform,
    qubits: Qubits,
) -> CHSHData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""

    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = CHSHData(bell_states=params.bell_states, thetas=thetas.tolist())

    if params.apply_error_mitigation:
        mitigation_data = mitigation_acquisition(
            mitigation_params(pulses=True, nshots=params.nshots), platform, qubits
        )
        mitigation_results = mitigation_fit(mitigation_data)

    for pair in qubits:
        if params.apply_error_mitigation:
            data.mitigation_matrix[pair] = mitigation_results.readout_mitigation_matrix[
                pair
            ]
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
                    frequencies = calculate_frequencies(results, list(pair))
                    data.register_basis(pair, bell_state, basis, frequencies)
    return data


def _acquisition_circuits(
    params: CHSHParameters,
    platform: Platform,
    qubits: Qubits,
) -> CHSHData:
    """Data acquisition for CHSH protocol using circuits."""
    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = CHSHData(
        bell_states=params.bell_states,
        thetas=thetas.tolist(),
    )

    if params.apply_error_mitigation:
        mitigation_data = mitigation_acquisition(
            mitigation_params(pulses=False, nshots=params.nshots), platform, qubits
        )
        mitigation_results = mitigation_fit(mitigation_data)
    for pair in qubits:
        if params.apply_error_mitigation:
            data.mitigation_matrix[pair] = mitigation_results.readout_mitigation_matrix[
                pair
            ]
        for bell_state in params.bell_states:
            for theta in thetas:
                chsh_circuits = create_chsh_circuits(
                    platform,
                    qubits=pair,
                    bell_state=bell_state,
                    theta=theta,
                    native=params.native,
                )
                for basis, circuit in chsh_circuits.items():
                    result = circuit(nshots=params.nshots)
                    frequencies = result.frequencies()
                    data.register_basis(pair, bell_state, basis, frequencies)

    return data


def _plot(data: CHSHData, fit: CHSHResults, qubit):
    """Plotting function for CHSH protocol."""
    figures = []

    for bell_state in data.bell_states:
        fig = go.Figure(layout_yaxis_range=[-3, 3])
        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=data.thetas,
                    y=fit.chsh[qubit[0], qubit[1], bell_state],
                    name="Bare",
                )
            )
            if fit.chsh_mitigated:
                fig.add_trace(
                    go.Scatter(
                        x=data.thetas,
                        y=fit.chsh_mitigated[qubit[0], qubit[1], bell_state],
                        name="Mitigated",
                    )
                )

        # classical bounds
        fig.add_hline(
            y=2,
            line_width=2,
            line_color="red",
        )
        fig.add_hline(
            y=-2,
            line_width=2,
            line_color="red",
        )

        # maximum values
        fig.add_hline(
            y=2 * np.sqrt(2),
            line_width=2,
            line_dash="dash",
            line_color="grey",
        )
        fig.add_hline(
            y=-2 * np.sqrt(2),
            line_width=2,
            line_dash="dash",
            line_color="grey",
        )

        fig.update_layout(
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Theta[rad]",
            yaxis_title="CHSH value",
        )
        figures.append(fig)

    return figures, None


def _fit(data: CHSHData) -> CHSHResults:
    """Fitting for CHSH protocol."""
    results = {}
    mitigated_results = {}
    for pair in data.pairs:
        for bell_state in data.bell_states:
            freq = data.merge_frequencies(pair, bell_state)
            if data.mitigation_matrix:
                matrix = data.mitigation_matrix[pair]

                mitigated_freq_list = []
                for freq_basis in freq:
                    mitigated_freq = {format(i, f"0{2}b"): [] for i in range(4)}
                    for i in range(len(data.thetas)):
                        freq_array = np.zeros(4)
                        for k, v in freq_basis.items():
                            freq_array[int(k, 2)] = v[i]
                        freq_array = freq_array.reshape(-1, 1)
                        for j, val in enumerate(matrix @ freq_array):
                            mitigated_freq[format(j, f"0{2}b")].append(float(val))
                    mitigated_freq_list.append(mitigated_freq)

            results[pair[0], pair[1], bell_state] = [
                compute_chsh(freq, bell_state, l) for l in range(len(data.thetas))
            ]

            if data.mitigation_matrix:
                mitigated_results[pair[0], pair[1], bell_state] = [
                    compute_chsh(mitigated_freq_list, bell_state, l)
                    for l in range(len(data.thetas))
                ]
    return CHSHResults(chsh=results, chsh_mitigated=mitigated_results)


chsh_circuits = Routine(_acquisition_circuits, _fit, _plot, two_qubit_gates=True)
"""CHSH experiment using circuits."""
chsh_pulses = Routine(_acquisition_pulses, _fit, _plot, two_qubit_gates=True)
"""CHSH experiment using pulses."""
