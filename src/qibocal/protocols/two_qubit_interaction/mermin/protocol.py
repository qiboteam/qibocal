import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo.backends import GlobalBackend
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit
from qibocal.config import log

from ...readout_mitigation_matrix import (
    ReadoutMitigationMatrixParameters as mitigation_params,
)
from ...readout_mitigation_matrix import _acquisition as mitigation_acquisition
from ...readout_mitigation_matrix import _fit as mitigation_fit

from ...readout_mitigation_matrix import readout_mitigation_matrix
from ...utils import calculate_frequencies
from .circuits import create_mermin_circuits
from .pulses import create_mermin_sequences
from .utils import (
    compute_mermin,
    get_mermin_coefficients,
    get_mermin_polynomial,
    get_readout_basis,
)

MITIGATION_MATRIX_FILE = "mitigation_matrix"
"""File where readout mitigation matrix is stored."""


@dataclass
class MerminParameters(Parameters):
    """Mermin experiment input parameters."""

    ntheta: int
    """Number of angles probed linearly between 0 and 2 pi."""
    native: Optional[bool] = False
    """If True a circuit will be created using only GPI2 and CZ gates."""
    apply_error_mitigation: Optional[bool] = False
    """Error mitigation model"""


@dataclass
class MerminData(Data):
    """Mermin Data structure."""

    thetas: list
    """Angles probed."""
    data: dict[list, list, str] = field(default_factory=dict)
    """Raw data acquired."""
    mitigation_matrix: dict[tuple[QubitId, ...], npt.NDArray] = field(
        default_factory=dict
    )
    targets=None
    """Mitigation matrix computed using the readout_mitigation_matrix protocol."""

    def save(self, path: Path):
        """Saving data including mitigation matrix."""
        pass
        # np.savez(
        #     path / f"{MITIGATION_MATRIX_FILE}.npz",
        #     self.mitigation_matrix
        #     **{
        #         json.dumps(tuple(targets)): self.mitigation_matrix[control, target]
        #         for control, target, _ in self.data
        #     },
        # )
        # super().save(path=path)
    

    @classmethod
    def load(cls, path: Path):
        """Custom loading to acco   modate mitigation matrix"""
        pass
        # instance = super().load(path=path)
        # # load readout mitigation matrix
        # mitigation_matrix = super().load_data(
        #     path=path, filename=MITIGATION_MATRIX_FILE
        # )
        # instance.mitigation_matrix = mitigation_matrix
        # return instance

    def register_basis(self, targets, basis, frequencies):
        """Store output for single qubit."""
        n = len(targets)
        self.targets = targets
        computational_basis = [format(i, f"0{n}b") for i in range(2**n)]

        # Add zero if state do not appear in state
        # could be removed by using high number of shots
        for i in computational_basis:
            if i not in frequencies:
                frequencies[i] = 0

        # print(basis)
        # print(frequencies.items())
        for state, freq in frequencies.items():
            # print(state, freq)
            # print(self.data)
            if ("".join(targets), basis, state) in self.data:
                self.data["".join(targets), basis, state] = np.concatenate(
                    (
                        self.data["".join(targets), basis, state],
                        np.array([freq]),
                    )
                )
            else:
                self.data["".join(targets), basis, state] = np.array([freq])

    def merge_frequencies(self, targets, readout_basis):
        """Merge frequencies with different measurement basis."""
        # mermin_polynomial = get_mermin_polynomial(len(targets))
        # readout_basis = get_readout_basis(mermin_polynomial)

        freqs = []
        mermin_data = {
            (index[1], index[2]): value
            for index, value in self.data.items()
            if index[0] == "".join(targets)
        }

        freqs = []
        for i in readout_basis:
            freqs.append(
                {
                    state[1]: value
                    for state, value in mermin_data.items()
                    if state[0] == i
                }
            )

        return freqs

    @property
    def params(self):
        """Convert non-arrays attributes into dict."""
        data_dict = super().params
        data_dict.pop("mitigation_matrix")

        return data_dict


@dataclass
class MerminResults(Results):
    """Mermin Results class."""

    mermin: dict[list, float] = field(default_factory=dict)
    """Raw Mermin value."""
    mermin_mitigated: dict[list, float] = field(default_factory=dict)
    """Mitigated Mermin value."""

    def __contains__(self, key: list):
        """Check if key is in class.

        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.

        """

        return key in [target for target in self.mermin]


def _acquisition_pulses(
    params: MerminParameters,
    platform: Platform,
    targets: list[QubitId],
) -> MerminData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""

    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = MerminData(thetas=thetas.tolist())
    # targets = list(targets[0])
    n = len(targets)
    # print(targets, n)
    mermin_polynomial = get_mermin_polynomial(n)
    readout_basis = get_readout_basis(mermin_polynomial)
    mermin_coefficients = get_mermin_coefficients(mermin_polynomial)

    if params.apply_error_mitigation:
        mitigation_data, _ = readout_mitigation_matrix.acquisition(
            readout_mitigation_matrix.parameters_type.load(
                dict(pulses=True, nshots=params.nshots)
            ),
            platform,
            [targets],
        )

        mitigation_results, _ = readout_mitigation_matrix.fit(mitigation_data)

    for theta in thetas:
        mermin_sequences = create_mermin_sequences(
            platform, targets, readout_basis=readout_basis, theta=theta
        )
        options = ExecutionParameters(nshots=params.nshots)

        mermin_frequencies = []
        for basis, sequence in mermin_sequences.items():
            results = platform.execute_pulse_sequence(sequence, options=options)
            frequencies = calculate_frequencies(results, targets)
            data.register_basis(targets, basis, frequencies)

    return data


def _acquisition_circuits(
    params: MerminParameters,
    platform: Platform,
    targets: list[tuple[QubitId]],
) -> MerminData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""

    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = MerminData(thetas=thetas.tolist())
    targets = list(targets[0])
    n = len(targets)
    mermin_polynomial = get_mermin_polynomial(n)
    readout_basis = get_readout_basis(mermin_polynomial)
    mermin_coefficients = get_mermin_coefficients(mermin_polynomial)

    if params.apply_error_mitigation:
        mitigation_data, _ = readout_mitigation_matrix.acquisition(
            readout_mitigation_matrix.parameters_type.load(
                dict(pulses=False, nshots=params.nshots)
            ),
            platform,
            [targets],
        )

        mitigation_results, _ = readout_mitigation_matrix.fit(mitigation_data)

    mermin_circuits = create_mermin_circuits(
        targets, native=params.native, readout_basis=readout_basis
    )

    for basis, circuit in mermin_circuits.items():
        results = circuit(nshots=params.nshots)
        data.register_basis(targets, basis, results.frequencies())

    # mermin_bare = compute_mermin(frequencies=mermin_frequencies, mermin_coefficients)

    return data


def _plot(data: MerminData, fit: MerminResults, target):
    """Plotting function for Mermin protocol."""
    figures = []
    targets = data.targets

    n = len(targets)
    classical_bound = 2 ** (n // 2)
    quantum_bound = 2 ** ((n - 1) / 2) * (2 ** (n // 2))

    fig = go.Figure(layout_yaxis_range=[-3, 3])
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data.thetas,
                y=fit.mermin["".join(data.targets)],  # TODO: FIX
                name="Bare",
            )
        )
        if fit.mermin_mitigated:
            fig.add_trace(
                go.Scatter(
                    x=data.thetas,
                    y=fit.mermin_mitigated["".join(data.targets)],  # TODO: FIX
                    name="Mitigated",
                )
            )

    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=data.thetas,
            y=[+classical_bound] * len(data.thetas),
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
            y=[-classical_bound] * len(data.thetas),
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
            y=[+quantum_bound] * len(data.thetas),
            line_color="gray",
            name="Quantum limit",
            legendgroup="quantum",
        )
    )

    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=data.thetas,
            y=[-quantum_bound] * len(data.thetas),
            line_color="gray",
            name="Quantum limit",
            legendgroup="quantum",
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Theta [rad]",
        yaxis_title="Mermin polynomial value",
        xaxis=dict(range=[min(data.thetas), max(data.thetas)]),
    )
    figures.append(fig)

    return figures, ""


def _fit(data: MerminData) -> MerminResults:
    """Fitting for CHSH protocol."""
    results = {}
    mitigated_results = {}
    
    n = len(data.targets)
    mermin_polynomial = get_mermin_polynomial(n)
    readout_basis = get_readout_basis(mermin_polynomial)
    mermin_coefficients = get_mermin_coefficients(mermin_polynomial)
    freq = data.merge_frequencies(data.targets, readout_basis)


    if data.mitigation_matrix:
        matrix = data.mitigation_matrix[pair]
        mitigated_freq_list = []
        for freq_basis in freq:
            mitigated_freq = {format(i, f"0{n}b"): [] for i in range(2**n)}
            for i in range(len(data.thetas)):
                freq_array = np.zeros(2**n)
                for k, v in freq_basis.items():
                    freq_array[int(k, 2)] = v[i]
                freq_array = freq_array.reshape(-1, 1)
                for j, val in enumerate(matrix @ freq_array):
                    mitigated_freq[format(j, f"0{n}b")].append(float(val))
            mitigated_freq_list.append(mitigated_freq)
    print(freq, mermin_coefficients)
    results["".join(data.targets)] = [
        compute_mermin(freq, mermin_coefficients, l) for l in range(len(data.thetas))
    ]

    if data.mitigation_matrix:
        mitigated_results["".join(data.targets)] = [
            compute_mermin(mitigated_freq_list, mermin_coefficients, l)
            for l in range(len(data.thetas))
        ]
    print(results)
    # print(type(results[0]))
    print(mitigated_results)
    return MerminResults(mermin=results, mermin_mitigated=mitigated_results)


mermin_pulses = Routine(_acquisition_pulses, _fit, _plot)
mermin_circuits = Routine(_acquisition_circuits, _fit, _plot)
