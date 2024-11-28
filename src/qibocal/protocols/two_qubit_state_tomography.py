import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import NumpyBackend, get_backend
from qibo.quantum_info import fidelity, partial_trace
from qibo.result import QuantumState
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import DATAFILE, Data, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit

from .state_tomography import StateTomographyParameters, plot_reconstruction
from .utils import table_dict, table_html

SINGLE_QUBIT_BASIS = ["X", "Y", "Z"]
TWO_QUBIT_BASIS = list(product(SINGLE_QUBIT_BASIS, SINGLE_QUBIT_BASIS))
OUTCOMES = ["00", "01", "10", "11"]
SIMULATED_DENSITY_MATRIX = "ideal"
"""Filename for simulated density matrix."""


TomographyType = np.dtype(
    [("frequencies", np.int64), ("simulation_probabilities", np.float64)]
)
"""Custom dtype for tomography."""


@dataclass
class StateTomographyData(Data):
    """Tomography data."""

    data: dict[tuple[QubitId, QubitId, str, str], np.int64] = field(
        default_factory=dict
    )
    ideal: dict[QubitPairId, np.ndarray] = field(default_factory=dict)
    simulated: Optional[QuantumState] = None

    def save(self, path):
        self._to_npz(path, DATAFILE)
        np.savez(
            path / f"{SIMULATED_DENSITY_MATRIX}.npz",
            **{json.dumps(pair): rho for pair, rho in self.ideal.items()},
        )
        self.simulated.dump(path / "simulated.json")

    @classmethod
    def load(cls, path):
        return cls(
            data=super().load_data(path, DATAFILE),
            ideal=super().load_data(path, SIMULATED_DENSITY_MATRIX),
            simulated=QuantumState.load(path / "simulated.json"),
        )


@dataclass
class StateTomographyResults(Results):
    """Tomography results."""

    measured_raw_density_matrix_real: dict[QubitPairId, list] = field(
        default_factory=dict
    )
    """Real part of measured density matrix before projecting."""
    measured_raw_density_matrix_imag: dict[QubitPairId, list] = field(
        default_factory=dict
    )
    """Imaginary part of measured density matrix before projecting."""

    measured_density_matrix_real: dict[QubitPairId, list] = field(default_factory=dict)
    """Real part of measured density matrix after projecting."""
    measured_density_matrix_imag: dict[QubitPairId, list] = field(default_factory=dict)
    """Imaginary part of measured density matrix after projecting."""

    fidelity: dict[QubitId, float] = field(default_factory=dict)
    """State fidelity."""


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[QubitPairId]
) -> StateTomographyData:
    """Acquisition protocol for two qubit state tomography experiment."""
    qubits = [q for pair in targets for q in pair]
    for q, counts in Counter(qubits).items():
        if counts > 1:
            raise ValueError(
                f"Qubits can only be measured once, but qubit {q} is measured {counts} times."
            )

    if params.circuit is None:
        params.circuit = Circuit(len(qubits))

    backend = get_backend()
    backend.platform = platform
    simulator = NumpyBackend()
    transpiler = dummy_transpiler(backend)

    simulated_state = simulator.execute_circuit(deepcopy(params.circuit))
    data = StateTomographyData(simulated=simulated_state)
    for basis1, basis2 in TWO_QUBIT_BASIS:
        basis_circuit = deepcopy(params.circuit)
        # FIXME: https://github.com/qiboteam/qibo/issues/1318
        if basis1 != "Z":
            basis_circuit.add(
                getattr(gates, basis1)(2 * i).basis_rotation()
                for i in range(len(targets))
            )
        if basis2 != "Z":
            basis_circuit.add(
                getattr(gates, basis2)(2 * i + 1).basis_rotation()
                for i in range(len(targets))
            )

        basis_circuit.add(
            gates.M(2 * i, 2 * i + 1, register_name=f"reg{i}")
            for i in range(len(targets))
        )

        simulation_result = simulator.execute_circuit(basis_circuit)
        _, results = execute_transpiled_circuit(
            basis_circuit,
            qubits,
            backend,
            nshots=params.nshots,
            transpiler=transpiler,
        )

        for i, pair in enumerate(targets):
            frequencies = results.frequencies(registers=True)[f"reg{i}"]
            simulation_probabilities = simulation_result.probabilities(
                qubits=(2 * i, 2 * i + 1)
            )
            data.register_qubit(
                TomographyType,
                pair + (basis1, basis2),
                {
                    "frequencies": np.array([frequencies[i] for i in OUTCOMES]),
                    "simulation_probabilities": simulation_probabilities,
                },
            )
            if basis1 == "Z" and basis2 == "Z":
                nqubits = basis_circuit.nqubits
                traced_qubits = tuple(
                    q for q in range(nqubits) if q not in (2 * i, 2 * i + 1)
                )
                data.ideal[pair] = partial_trace(
                    simulation_result.state(), traced_qubits
                )

    return data


def rotation_matrix(basis):
    """Matrix of the gate implementing the rotation to the given basis.

    Args:
        basis (str): One of Pauli basis: X, Y or Z.
    """
    backend = NumpyBackend()
    if basis == "Z":
        return np.eye(2, dtype=complex)
    return getattr(gates, basis)(0).basis_rotation().matrix(backend)


def project_psd(matrix):
    """Project matrix to the space of positive semidefinite matrices."""
    s, v = np.linalg.eigh(matrix)
    s = s * (s > 0)
    return v.dot(np.diag(s)).dot(v.conj().T)


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for two qubit state tomography.

    Uses a linear inversion algorithm to reconstruct the density matrix
    from measurements, with the following steps:
    1. Construct a linear transformation M, from density matrix
    to Born-probabilities in the space of all two-qubit measurement bases
    (in our case XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ).
    2. Invert M to get the transformation from Born-probabilities to
    density matrices.
    3. Calculate vector of Born-probabilities from experimental measurements (frequencies).
    4. Map this vector to a density matrix (``measured_raw_density_matrix``) using the
    inverse of M from step 2.
    5. Project the calculated density matrix to the space of positive semidefinite
    matrices (``measured_density_matrix``) using the function ``project_psd``.
    """
    rotations = [
        np.kron(rotation_matrix(basis1), rotation_matrix(basis2))
        for basis1, basis2 in TWO_QUBIT_BASIS
    ]

    # construct the linear transformation from density matrix to Born-probabilities
    measurement = np.zeros((0, 16))
    for rotation in rotations:
        channel = np.kron(rotation, np.conj(rotation))
        measure_channel = channel[np.eye(4, dtype=bool).flatten(), :]
        measurement = np.concatenate((measurement, measure_channel))

    # invert to get linear transformation from Born-probabilities to density matrix
    inverse_measurement = np.linalg.pinv(measurement)

    # calculate Born-probabilities vector from measurements (frequencies)
    probabilities = defaultdict(lambda: np.empty(measurement.shape[0]))
    for (qubit1, qubit2, basis1, basis2), value in data.data.items():
        frequencies = value["frequencies"]
        ib = TWO_QUBIT_BASIS.index((basis1, basis2))
        probabilities[(qubit1, qubit2)][4 * ib : 4 * (ib + 1)] = frequencies / np.sum(
            frequencies
        )

    # reconstruction
    backend = NumpyBackend()
    results = StateTomographyResults()
    for pair, probs in probabilities.items():
        measured_rho = inverse_measurement.dot(probs).reshape((4, 4))
        measured_rho_proj = project_psd(measured_rho)

        results.measured_raw_density_matrix_real[pair] = measured_rho.real.tolist()
        results.measured_raw_density_matrix_imag[pair] = measured_rho.imag.tolist()
        results.measured_density_matrix_real[pair] = measured_rho_proj.real.tolist()
        results.measured_density_matrix_imag[pair] = measured_rho_proj.imag.tolist()
        results.fidelity[pair] = fidelity(measured_rho_proj, data.ideal[pair])

    return results


def plot_measurements(data: StateTomographyData, target: QubitPairId):
    """Plot histogram of measurements in the 9 different basis."""
    qubit1, qubit2 = target
    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=tuple(
            f"{basis1}<sub>1</sub>{basis2}<sub>2</sub>"
            for basis1, basis2 in TWO_QUBIT_BASIS
        ),
    )
    for i, (basis1, basis2) in enumerate(TWO_QUBIT_BASIS):
        row = i // 3 + 1
        col = i % 3 + 1
        basis_data = data.data[qubit1, qubit2, basis1, basis2]

        fig.add_trace(
            go.Bar(
                x=OUTCOMES,
                y=basis_data["simulation_probabilities"],
                name="simulation",
                width=0.5,
                marker_color=color1,
                legendgroup="simulation",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

        frequencies = basis_data["frequencies"]
        fig.add_trace(
            go.Bar(
                x=OUTCOMES,
                y=frequencies / np.sum(frequencies),
                name="experiment",
                width=0.5,
                marker_color=color2,
                legendgroup="experiment",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

    fig.update_yaxes(range=[0, 1])
    fig.update_layout(barmode="overlay")

    return fig


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitPairId):
    """Plotting for two qubit state tomography."""
    if isinstance(target, list):
        target = tuple(target)

    fig_measurements = plot_measurements(data, target)
    if fit is None:
        fitting_report = table_html(
            table_dict(
                [target],
                ["Target state"],
                [str(data.simulated)],
            )
        )
        return [fig_measurements], fitting_report

    measured = np.array(fit.measured_density_matrix_real[target]) + 1j * np.array(
        fit.measured_density_matrix_imag[target]
    )
    fig = plot_reconstruction(data.ideal[target], measured)

    fitting_report = table_html(
        table_dict(
            [target, target],
            ["Target state", "Fidelity"],
            [str(data.simulated), np.round(fit.fidelity[target], 4)],
        )
    )

    return [fig_measurements, fig], fitting_report


two_qubit_state_tomography = Routine(_acquisition, _fit, _plot)
