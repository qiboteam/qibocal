import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import GlobalBackend, NumpyBackend
from qibo.quantum_info import fidelity
from qibo.result import QuantumState
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import DATAFILE, Data, Parameters, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit

from .utils import table_dict, table_html

PAULI_BASIS = ["X", "Y", "Z"]
OUTCOMES = ["00", "01", "10", "11"]
SIMULATED_DENSITY_MATRIX = "ideal"
"""Filename for simulated density matrix."""


@dataclass
class StateTomographyParameters(Parameters):
    """Tomography input parameters"""

    circuit: Optional[Union[str, Circuit]] = None
    """Circuit to prepare initial state.

        It can also be provided the path to a json file containing
        a serialized circuit.
    """

    def __post_init__(self):
        if isinstance(self.circuit, str):
            raw = json.loads((Path.cwd() / self.circuit).read_text())
            self.circuit = Circuit.from_dict(raw)


TomographyType = np.dtype(
    [("frequencies", np.int64), ("simulation_probabilities", np.float64)]
)
"""Custom dtype for tomography."""


@dataclass
class StateTomographyData(Data):
    """Tomography data"""

    data: dict[tuple[QubitId, str, QubitId, str], np.int64] = field(
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
    """Tomography results"""

    measured_density_matrix_real: dict[QubitPairId, list]
    """Real part of measured density matrix."""
    measured_density_matrix_imag: dict[QubitPairId, list]
    """Imaginary part of measured density matrix."""
    measured_density_matrix_projected_real: dict[QubitPairId, list]
    """Real part of measured density matrix."""
    measured_density_matrix_projected_imag: dict[QubitPairId, list]
    """Imaginary part of measured density matrix."""
    fidelity: dict[QubitId, float]
    """State fidelity."""


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[QubitPairId]
) -> StateTomographyData:
    """Acquisition protocol for single qubit state tomography experiment."""
    qubits = [q for pair in targets for q in pair]
    for q, counts in Counter(qubits).items():
        if counts > 1:
            raise ValueError(
                f"Qubits can only be measured once, but qubit {q} is measured {counts} times."
            )

    if params.circuit is None:
        params.circuit = Circuit(len(qubits))

    backend = GlobalBackend()
    simulator = NumpyBackend()
    transpiler = dummy_transpiler(backend)

    simulated_state = simulator.execute_circuit(deepcopy(params.circuit))
    data = StateTomographyData(simulated=simulated_state)
    for basis1, basis2 in product(PAULI_BASIS, PAULI_BASIS):
        basis_circuit = deepcopy(params.circuit)
        # FIXME: basis
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

        for i, (q1, q2) in enumerate(targets):
            frequencies = results.frequencies(registers=True)[f"reg{i}"]
            simulation_probabilities = simulation_result.probabilities(
                qubits=(2 * i, 2 * i + 1)
            )
            data.register_qubit(
                TomographyType,
                (q1, basis1, q2, basis2),
                {
                    "frequencies": np.array([frequencies[i] for i in OUTCOMES]),
                    "simulation_probabilities": simulation_probabilities,
                },
            )
            if basis1 == "Z" and basis2 == "Z":
                nqubits = basis_circuit.nqubits
                statevector = simulation_result.state()
                data.ideal[(q1, q2)] = simulator.partial_trace(
                    statevector, (2 * i, 2 * i + 1), nqubits
                )

    return data


def rotation_matrix(basis):
    backend = NumpyBackend()
    if basis == "Z":
        return np.eye(2, dtype=complex)
    return getattr(gates, basis)(0).basis_rotation().matrix(backend)


def project_psd(matrix):
    s, v = np.linalg.eigh(matrix)
    s = s * (s > 0)
    return v.dot(np.diag(s)).dot(v.conj().T)


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""
    basis_list = list(product(PAULI_BASIS, PAULI_BASIS))
    rotations = [
        np.kron(rotation_matrix(basis1), rotation_matrix(basis2))
        for basis1, basis2 in basis_list
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
    for (qubit1, basis1, qubit2, basis2), value in data.data.items():
        frequencies = value["frequencies"]
        ib = basis_list.index((basis1, basis2))
        probabilities[(qubit1, qubit2)][4 * ib : 4 * (ib + 1)] = frequencies / np.sum(
            frequencies
        )

    # reconstruction
    backend = NumpyBackend()
    results = StateTomographyResults({}, {}, {}, {}, {})
    for pair, probs in probabilities.items():
        measured_rho = inverse_measurement.dot(probs).reshape((4, 4))
        measured_rho_proj = project_psd(measured_rho)
        target_rho = backend.partial_trace(
            data.simulated.state(), pair, data.simulated.nqubits
        )

        results.measured_density_matrix_real[pair] = measured_rho.real.tolist()
        results.measured_density_matrix_imag[pair] = measured_rho.imag.tolist()
        results.measured_density_matrix_projected_real[pair] = (
            measured_rho_proj.real.tolist()
        )
        results.measured_density_matrix_projected_imag[pair] = (
            measured_rho_proj.imag.tolist()
        )
        results.fidelity[pair] = fidelity(measured_rho_proj, data.ideal[pair])

    return results


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitPairId):
    """Plotting for state tomography"""
    qubit1, qubit2 = target
    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"

    fig1 = make_subplots(
        rows=3,
        cols=3,
        # "$\text{Plot 1}$"
        subplot_titles=tuple(
            f"{basis1}<sub>1</sub>{basis2}<sub>2</sub>"
            for basis1, basis2 in product(PAULI_BASIS, PAULI_BASIS)
        ),
    )
    for i, (basis1, basis2) in enumerate(product(PAULI_BASIS, PAULI_BASIS)):
        row = i // 3 + 1
        col = i % 3 + 1
        basis_data = data.data[qubit1, basis1, qubit2, basis2]

        fig1.add_trace(
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
        fig1.add_trace(
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

    fig1.update_yaxes(range=[0, 1])
    fig1.update_layout(barmode="overlay")  # , height=900)

    fitting_report = table_html(
        table_dict(
            [target],
            ["Target state"],
            [str(data.simulated)],
        )
    )

    if fit is not None:
        fig2 = make_subplots(
            rows=2,
            cols=2,
            # "$\text{Plot 1}$"
            subplot_titles=(
                "Re(ρ)<sub>measured</sub>",
                "Im(ρ)<sub>measured</sub>",
                "Re(ρ)<sub>theory</sub>",
                "Im(ρ)<sub>theory</sub>",
            ),
        )
        # computing limits for colorscale
        min_re, max_re = np.min(
            fit.measured_density_matrix_projected_real[target]
        ), np.max(fit.measured_density_matrix_projected_real[target])
        min_im, max_im = np.min(
            fit.measured_density_matrix_projected_imag[target]
        ), np.max(fit.measured_density_matrix_projected_imag[target])

        # add offset
        if np.abs(min_re - max_re) < 1e-5:
            min_re = min_re - 0.1
            max_re = max_re + 0.1

        if np.abs(min_im - max_im) < 1e-5:
            min_im = min_im - 0.1
            max_im = max_im + 0.1
        fig2.add_trace(
            go.Heatmap(
                z=fit.measured_density_matrix_projected_real[target],
                x=OUTCOMES,
                y=OUTCOMES,
                colorscale="ice",
                colorbar_x=-0.2,
                zmin=min_re,
                zmax=max_re,
                reversescale=True,
            ),
            row=1,
            col=1,
        )

        fig2.add_trace(
            go.Heatmap(
                z=data.ideal[target].real,
                x=OUTCOMES,
                y=OUTCOMES,
                showscale=False,
                colorscale="ice",
                zmin=min_re,
                zmax=max_re,
                reversescale=True,
            ),
            row=2,
            col=1,
        )

        fig2.add_trace(
            go.Heatmap(
                z=fit.measured_density_matrix_projected_imag[target],
                x=OUTCOMES,
                y=OUTCOMES,
                colorscale="Burg",
                colorbar_x=1.01,
                zmin=min_im,
                zmax=max_im,
            ),
            row=1,
            col=2,
        )

        fig2.add_trace(
            go.Heatmap(
                z=data.ideal[target].imag,
                x=OUTCOMES,
                y=OUTCOMES,
                colorscale="Burg",
                showscale=False,
                zmin=min_im,
                zmax=max_im,
            ),
            row=2,
            col=2,
        )
        fitting_report = table_html(
            table_dict(
                [target, target],
                ["Target state", "Fidelity"],
                [str(data.simulated), np.round(fit.fidelity[target], 4)],
            )
        )

        return [fig1, fig2], fitting_report

    return [fig1], fitting_report


two_qubit_state_tomography = Routine(_acquisition, _fit, _plot)
