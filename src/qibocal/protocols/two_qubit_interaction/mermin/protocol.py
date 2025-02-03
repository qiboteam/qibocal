from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from ...readout_mitigation_matrix import readout_mitigation_matrix
from ...utils import STRING_TYPE, calculate_frequencies
from .pulses import create_mermin_sequences
from .utils import (
    compute_mermin,
    get_mermin_coefficients,
    get_mermin_polynomial,
    get_readout_basis,
)

PLOT_PADDING = 0.2


@dataclass
class MerminParameters(Parameters):
    """Mermin experiment input parameters."""

    ntheta: int
    """Number of angles probed linearly between 0 and 2 pi."""
    native: Optional[bool] = False
    """If True a circuit will be created using only GPI2 and CZ gates."""
    apply_error_mitigation: Optional[bool] = False
    """Error mitigation model"""


MerminType = np.dtype(
    [
        ("theta", float),
        ("basis", STRING_TYPE),
        ("state", int),
        ("frequency", int),
    ]
)


@dataclass
class MerminData(Data):
    """Mermin Data structure."""

    thetas: list
    """Angles probed."""
    data: dict[list[QubitId], npt.NDArray[MerminType]] = field(default_factory=dict)
    """Raw data acquired."""
    mitigation_matrix: dict[list[QubitId], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Mitigation matrix computed using the readout_mitigation_matrix protocol."""

    @property
    def targets(self):
        return list(self.data)


@dataclass
class MerminResults(Results):
    """Mermin Results class."""

    mermin: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Raw Mermin value."""

    mermin_mitigated: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Mitigated Mermin value."""


def _acquisition(
    params: MerminParameters,
    platform: Platform,
    targets: list[list[QubitId]],
) -> MerminData:
    """Data acquisition for Mermin protocol using pulse sequences."""

    thetas = np.linspace(0, 2 * np.pi, params.ntheta)
    data = MerminData(thetas=thetas.tolist())
    if params.apply_error_mitigation:
        mitigation_data, _ = readout_mitigation_matrix.acquisition(
            readout_mitigation_matrix.parameters_type.load(dict(nshots=params.nshots)),
            platform,
            targets,
        )

        mitigation_results, _ = readout_mitigation_matrix.fit(mitigation_data)
        data.mitigation_matrix = mitigation_results.readout_mitigation_matrix
    platform.connect()
    for qubits in targets:
        mermin_polynomial = get_mermin_polynomial(len(qubits))
        readout_basis = get_readout_basis(mermin_polynomial)

        for theta in thetas:
            mermin_sequences = create_mermin_sequences(
                platform, qubits, readout_basis=readout_basis, theta=theta
            )
            options = ExecutionParameters(nshots=params.nshots)
            # TODO: use unrolling
            for basis, sequence in mermin_sequences.items():
                results = platform.execute_pulse_sequence(sequence, options=options)
                frequencies = calculate_frequencies(results, qubits)
                for state, frequency in enumerate(frequencies.values()):
                    data.register_qubit(
                        MerminType,
                        tuple(qubits),
                        dict(
                            theta=np.array([theta]),
                            basis=np.array([basis]),
                            state=np.array([state]),
                            frequency=np.array([frequency]),
                        ),
                    )
    return data


def _fit(data: MerminData) -> MerminResults:
    """Fitting for Mermin protocol."""
    targets = data.targets
    results = {qubits: [] for qubits in targets}
    mitigated_results = {qubits: [] for qubits in targets}
    basis = np.unique(data.data[targets[0]].basis)
    for qubits in targets:
        mermin_polynomial = get_mermin_polynomial(len(qubits))
        mermin_coefficients = get_mermin_coefficients(mermin_polynomial)

        for theta in data.thetas:
            qubit_data = data.data[qubits]
            outputs = []
            mitigated_outputs = []
            for base in basis:
                frequencies = np.zeros(2 ** len(qubits))
                data_filter = (qubit_data.basis == base) & (qubit_data.theta == theta)
                filtered_data = qubit_data[data_filter]
                state_freq = qubit_data[data_filter].frequency
                for state, freq in zip(filtered_data.state, filtered_data.frequency):
                    frequencies[state] = freq

                outputs.append(
                    {
                        format(i, f"0{len(qubits)}b"): freq
                        for i, freq in enumerate(state_freq)
                    }
                )

                if data.mitigation_matrix:
                    mitigated_output = np.dot(
                        data.mitigation_matrix[qubits],
                        frequencies,
                    )
                    mitigated_outputs.append(
                        {
                            format(i, f"0{len(qubits)}b"): freq
                            for i, freq in enumerate(mitigated_output)
                        }
                    )
            if data.mitigation_matrix:
                mitigated_results[tuple(qubits)].append(
                    compute_mermin(mitigated_outputs, mermin_coefficients)
                )
            results[tuple(qubits)].append(compute_mermin(outputs, mermin_coefficients))
    return MerminResults(
        mermin=results,
        mermin_mitigated=mitigated_results,
    )


def _plot(data: MerminData, fit: MerminResults, target):
    """Plotting function for Mermin protocol."""
    figures = []

    n_qubits = len(target)
    classical_bound = 2 ** (n_qubits // 2)
    quantum_bound = 2 ** ((n_qubits - 1) / 2) * (2 ** (n_qubits // 2))

    fig = go.Figure(
        layout_yaxis_range=[-quantum_bound - PLOT_PADDING, quantum_bound + PLOT_PADDING]
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data.thetas,
                y=fit.mermin[tuple(target)],
                name="Bare",
            )
        )
        if fit.mermin_mitigated:
            fig.add_trace(
                go.Scatter(
                    x=data.thetas,
                    y=fit.mermin_mitigated[tuple(target)],
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


mermin = Routine(_acquisition, _fit, _plot)
