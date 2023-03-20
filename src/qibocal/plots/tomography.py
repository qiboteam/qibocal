from functools import lru_cache
from typing import List

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibocal.data import DataUnits


def measurement_transformation(basis_matrices):
    """Construct the linear transformation from density matrix to Born-probabilities."""
    dim = len(basis_matrices[0])
    measurement = np.zeros((0, dim**2))
    for matrix in basis_matrices:
        channel = np.kron(matrix, matrix.conj())
        idx = np.eye(dim, dtype=bool).flatten()
        measurement = np.concatenate((measurement, channel[idx, :]))
    return measurement


class Rotations:
    OUTCOMES = ["00", "01", "10", "11"]

    def __init__(self, *rotations):
        self.gates = []
        self.labels = []
        for label, gate in rotations:
            self.labels.append(label)
            self.gates.append(gate)

        backend = NumpyBackend()
        self.matrices: List[np.ndarray] = []
        for gate in self.gates:
            if gate is None:
                self.matrices.append(np.eye(2, dtype=complex))
            else:
                self.matrices.append(gate(0).asmatrix(backend))

        self.transformation: np.ndarray = measurement_transformation(
            list(self.two_qubit_matrices())
        )
        self.inverse_transformation: np.ndarray = np.linalg.pinv(self.transformation)

    def two_qubit_matrices(self):
        """Yields the 4x4 matrices corresponding to all two-qubit pairs of the given rotations."""
        for u1 in self.matrices:
            for u2 in self.matrices:
                yield np.kron(u1, u2)

    def two_qubit_labels(self):
        """Yields the label tuples corresponding to all two-qubit pairs of the given rotations."""
        for label1 in self.labels:
            for label2 in self.labels:
                yield (label1, label2)

    def circuits(self, state_preperation):
        """Yields Qibo circuits that perform the state tomography.

        These circuits consist of the given ``state_preperation``
        circuit plus a rotation to a measurement basis.
        """
        for gate1 in self.gates:
            for gate2 in self.gates:
                c = Circuit(2)
                c.add(state_preperation.queue)
                if gate1 is not None:
                    c.add(gate1(0))
                if gate2 is not None:
                    c.add(gate2(1))
                c.add(gates.M(0, 1))
                yield c

    def simulate_probabilities_exact(self, state_preperation):
        """Simulate the tomography procedure and obtain exact probabilities."""
        backend = NumpyBackend()
        probs = np.empty(0)
        for c in self.circuits(state_preperation):
            result = backend.execute_circuit(c)
            probs = np.concatenate((probs, result.probabilities()))
        return probs

    def simulate_probabilities_measured(self, state_preperation, nshots):
        """Simulate the tomography procedure and obtain probabilities using simulated shots."""
        backend = NumpyBackend()
        probs = np.empty(0)
        for c in self.circuits(state_preperation):
            result = backend.execute_circuit(c, nshots=nshots)
            frequencies = result.frequencies()
            meas_probs = [frequencies[x] / nshots for x in self.OUTCOMES]
            probs = np.concatenate((probs, meas_probs))
        return probs

    def load_experiment_probabilities(self, folder, routine):
        """Load experiment data and calculate probabilities from single shots."""
        data = DataUnits.load_data(folder, "data", routine, "csv", "data").df
        probs = np.empty(0)
        for r1, r2 in self.two_qubit_labels():
            _data = data[(data["rotation1"] == r1) & (data["rotation2"] == r2)]
            probs = np.concatenate(
                (probs, [float(_data[f"probability{x}"]) for x in self.OUTCOMES])
            )
        return probs

    def reconstruct(self, data):
        return self.inverse_transformation.dot(data)


def project_to_density_matrix(A):
    """Project arbitrary matrix to a positive semi-definate normalized density matrix."""
    s, V = np.linalg.eigh(A)
    s = s * (s > 0)
    rho = V.dot(np.diag(s)).dot(V.conj().T)
    return rho / np.trace(rho)


def purity(rho):
    """Calculates the purity of a density matrix."""
    return np.trace(rho @ rho).real


class DensityMatrix:
    def __init__(self, rho):
        self.direct = rho
        self.projected = project_to_density_matrix(rho)

    @classmethod
    def reconstruct(cls, rotations, data):
        rho = rotations.reconstruct(data)
        return cls(np.reshape(rho, (4, 4)))

    def round(self, decimals=5):
        return np.round(self.projected, decimals=decimals)

    def projection_error(self):
        return np.linalg.norm(self.projected - self.direct, ord="fro")

    def purity(self):
        return purity(self.projected)

    def reduced1(self):
        return np.trace(self.projected.reshape(2, 2, 2, 2), axis1=0, axis2=2)

    def purity1(self):
        return purity(self.reduced1())

    def reduced2(self):
        return np.trace(self.projected.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    def purity2(self):
        return purity(self.reduced2())


def circuit_from_sequence(folder, routine, rotations):
    """Create a qibo circuit that simulates the tomography procedure."""
    basis_gates = {
        "I": None,
        "RX": lambda q: gates.RX(q, theta=np.pi),
        "RY": lambda q: gates.RY(q, theta=np.pi),
        "RX90": lambda q: gates.RX(q, theta=np.pi / 2),
        "RY90": lambda q: gates.RY(q, theta=np.pi / 2),
    }

    with open(f"{folder}/runcard.yml") as file:
        action_runcard = yaml.safe_load(file)

    experiment_qubits = action_runcard["qubits"]
    sequence = action_runcard["actions"][routine]["sequence"]
    nshots = action_runcard["actions"][routine]["nshots"]

    circuit = Circuit(2)
    for moment in sequence:
        for pulse_description in moment:
            pulse_type, qubit = pulse_description[:2]
            if pulse_type == "FluxPulse":
                # FIXME: FluxPulse is not always one-to-one to CZ gate
                circuit.add(gates.CZ(0, 1))
            else:
                circuit.add(basis_gates[pulse_type](experiment_qubits.index(qubit)))
    return circuit, nshots


@lru_cache(maxsize=None)
def prepare_plots(folder, routine):
    rotations = Rotations(
        ("I", None),
        ("RY90", lambda q: gates.RY(q, theta=np.pi / 2)),
        ("RX90", lambda q: gates.RX(q, theta=np.pi / 2)),
    )
    circuit, _ = circuit_from_sequence(folder, routine, rotations)
    simulation_probabilities = rotations.simulate_probabilities_exact(circuit)
    experiment_probabilities = rotations.load_experiment_probabilities(folder, routine)
    return rotations, circuit, simulation_probabilities, experiment_probabilities


def probabilities_bar_chart(folder, routine, qubit, data_format):
    """Generates the probability bar chart for each measurement basis for the report."""
    (
        rotations,
        circuit,
        simulation_probabilities,
        experiment_probabilities,
    ) = prepare_plots(folder, routine)

    labels = ["00", "01", "10", "11"]
    titles = [f"({r1}, {r2})" for r1, r2 in rotations.two_qubit_labels()]
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles)
    row, col = 1, 1
    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"
    for i in range(len(simulation_probabilities) // 4):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=simulation_probabilities[4 * i : 4 * i + 4],
                name="simulation",
                width=0.5,
                marker_color=color1,
                legendgroup="simulation",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=experiment_probabilities[4 * i : 4 * i + 4],
                name="experiment",
                width=0.5,
                marker_color=color2,
                legendgroup="experiment",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

        col += 1
        if col > 5:
            row += 1
            col = 1

    fig.update_yaxes(range=[0, 1])
    fig.update_layout(barmode="overlay", height=900)

    fitting_report = []
    # circuit draw
    for circuit_line in circuit.draw().split("\n"):
        qubit_str, gate_str = circuit_line.split(":")
        fitting_report.append(f"{qubit_str} | {gate_str}: <br>")

    return [fig], "".join(fitting_report)


def density_matrix_reconstruction(folder, routine, qubit, data_format):
    (
        rotations,
        circuit,
        simulation_probabilities,
        experiment_probabilities,
    ) = prepare_plots(folder, routine)

    rho_experiment = DensityMatrix.reconstruct(rotations, experiment_probabilities)
    rho_simulation = DensityMatrix.reconstruct(rotations, simulation_probabilities)

    labels = [f"<{format(i, '02b')}|" for i in range(4)]
    titles = [f"Re |{format(i, '02b')}>" for i in range(4)]
    titles.extend([f"Im |{format(i, '02b')}>" for i in range(4)])
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles)
    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"
    for row, comp in enumerate(["real", "imag"]):
        for col in range(4):
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=getattr(rho_simulation.round()[col], comp),
                    name="simulation",
                    width=0.5,
                    marker_color=color1,
                    legendgroup="simulation",
                    showlegend=row == 0 and col == 0,
                ),
                row=row + 1,
                col=col + 1,
            )
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=getattr(rho_experiment.round()[col], comp),
                    name="experiment",
                    width=0.5,
                    marker_color=color2,
                    legendgroup="experiment",
                    showlegend=row == 0 and col == 0,
                ),
                row=row + 1,
                col=col + 1,
            )

    # max_value = np.max(np.abs(rho_simulation.round()))
    # fig.update_yaxes(range=[-max_value, max_value])
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(barmode="overlay", height=900)

    fitting_report = []
    # circuit draw
    for circuit_line in circuit.draw().split("\n"):
        qubit_str, gate_str = circuit_line.split(":")
        fitting_report.append(f"{qubit_str} | {gate_str}: <br>")
    # reconstruction parameters
    error_from_sim = np.linalg.norm(
        rho_experiment.projected - rho_simulation.projected, ord="fro"
    )
    fitting_report.extend(
        [
            f"q{qubit}/r0 | Projection error: {rho_experiment.projection_error()}<br>",
            f"q{qubit}/r0 | Error from simulation: {error_from_sim}<br>",
            f"q{qubit}/r0 | Purity: {rho_experiment.purity()}<br>",
            f"q{qubit}/r0 | Purity (sim): {rho_simulation.purity()}<br>",
            f"q{qubit}/r0 | Reduced purity A: {rho_experiment.purity1()}<br>"
            f"q{qubit}/r0 | Reduced purity A (sim): {rho_simulation.purity1()}<br>"
            f"q{qubit}/r0 | Reduced purity B: {rho_experiment.purity2()}<br>",
            f"q{qubit}/r0 | Reduced purity B (sim): {rho_simulation.purity2()}<br>",
        ]
    )

    return [fig], "".join(fitting_report)
