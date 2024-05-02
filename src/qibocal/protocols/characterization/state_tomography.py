from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import NumpyBackend, matrices
from qibo.quantum_info import fidelity
from qibo.result import QuantumState
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import DATAFILE, Data, Parameters, Results, Routine

from .utils import table_dict, table_html


@dataclass
class StateTomographyParameters(Parameters):
    """Tomography input parameters"""

    circuit: Optional[Circuit] = None
    """Circuit to prepare initial state"""


TomographyType = np.dtype(
    [
        ("samples", np.int64),
    ]
)
"""Custom dtype for tomography."""


@dataclass
class StateTomographyData(Data):
    """Tomography data"""

    x_basis_state: QuantumState = None
    y_basis_state: QuantumState = None
    z_basis_state: QuantumState = None
    data: dict[tuple[QubitId, str], np.float64] = field(default_factory=dict)

    def save(self, path):
        self.x_basis_state.dump(path / "x.json")
        self.y_basis_state.dump(path / "y.json")
        self.z_basis_state.dump(path / "z.json")
        self._to_npz(path, DATAFILE)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.data = super().load_data(path, DATAFILE)
        instance.x_basis_state = QuantumState.load(path / "x.json")
        instance.y_basis_state = QuantumState.load(path / "y.json")
        instance.z_basis_state = QuantumState.load(path / "z.json")

        return instance


@dataclass
class StateTomographyResults(Results):
    """Tomography results"""

    density_matrix_real: dict[QubitId, list]
    density_matrix_imag: dict[QubitId, list]
    target_density_matrix_real: list
    target_density_matrix_imag: list
    fidelity: dict[QubitId, float]


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[QubitId]
) -> StateTomographyData:
    """Acquisition protocol for state tomography."""

    if params.circuit is None:
        params.circuit = Circuit(len(targets), wire_names=[str(i) for i in targets])

    data = StateTomographyData()

    for basis in ["X", "Y", "Z"]:
        basis_circuit = deepcopy(params.circuit)
        # FIXME: basis
        if basis != "Z":
            for i in range(len(targets)):
                basis_circuit.add(getattr(gates, basis)(i).basis_rotation())

        # basis_circuit.add(gates.M(i, basis=getattr(gates, basis)) for i in targets)
        for i in range(len(targets)):
            basis_circuit.add(gates.M(i))
        for i, target in enumerate(targets):
            data.register_qubit(
                TomographyType,
                (target, basis),
                dict(
                    samples=basis_circuit(nshots=params.nshots).samples(),
                ),
            )
            setattr(
                data,
                f"{basis.lower()}_basis_state",
                NumpyBackend().execute_circuit(basis_circuit),
            )
    return data


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""
    density_matrix_real = {}
    density_matrix_imag = {}
    fid = {}
    for qubit in data.qubits:
        x_exp, y_exp, z_exp = (
            1 - 2 * np.mean(data[qubit, basis].samples) for basis in ["X", "Y", "Z"]
        )
        density_matrix = 0.5 * (
            matrices.I + matrices.X * x_exp + matrices.Y * y_exp + matrices.Z * z_exp
        )
        density_matrix_real[qubit] = np.real(density_matrix).tolist()
        density_matrix_imag[qubit] = np.imag(density_matrix).tolist()

    x_theory = 1 - 2 * data.x_basis_state.probabilities([0])[1]
    y_theory = 1 - 2 * data.y_basis_state.probabilities([0])[1]
    z_theory = 1 - 2 * data.z_basis_state.probabilities([0])[1]
    target_density_matrix = 0.5 * (
        matrices.I
        + matrices.X * x_theory
        + matrices.Y * y_theory
        + matrices.Z * z_theory
    )
    target_density_matrix_real = np.real(target_density_matrix).tolist()
    target_density_matrix_imag = np.imag(target_density_matrix).tolist()

    for qubit in data.qubits:
        fid[qubit] = fidelity(
            np.array(density_matrix_real[qubit])
            + 1.0j * np.array(density_matrix_imag[qubit]),
            target_density_matrix,
        )

    return StateTomographyResults(
        density_matrix_real=density_matrix_real,
        density_matrix_imag=density_matrix_imag,
        target_density_matrix_real=target_density_matrix_real,
        target_density_matrix_imag=target_density_matrix_imag,
        fidelity=fid,
    )


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitId):
    """Plotting for state tomography"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Re(rho) exp",
            "Im(rho) exp",
            "Re(rho) th",
            "Im(rho) th",
        ),
    )

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                z=fit.density_matrix_real[target],
                x=["0", "1"],
                y=["0", "1"],
                hoverongaps=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.density_matrix_imag[target],
                x=["0", "1"],
                y=["0", "1"],
                hoverongaps=False,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.target_density_matrix_real,
                x=["0", "1"],
                y=["0", "1"],
                hoverongaps=False,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.target_density_matrix_imag,
                x=["0", "1"],
                y=["0", "1"],
                hoverongaps=False,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Fidelity",
                ],
                [
                    np.round(fit.fidelity[target], 4),
                ],
            )
        )

        return [fig], fitting_report
    return [], ""


state_tomography = Routine(_acquisition, _fit, _plot)
