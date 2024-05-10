import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import GlobalBackend, NumpyBackend, matrices
from qibo.quantum_info import fidelity
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import DATAFILE, Data, Parameters, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit

from .utils import table_dict, table_html

BASIS = ["X", "Y", "Z"]
"""Single qubit measurement basis."""
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
    [
        ("samples", np.int64),
    ]
)
"""Custom dtype for tomography."""


@dataclass
class StateTomographyData(Data):
    """Tomography data"""

    ideal: dict[tuple[QubitId, str], np.float64] = field(default_factory=dict)
    """Ideal samples measurements."""
    data: dict[tuple[QubitId, str], np.int64] = field(default_factory=dict)
    """Hardware measurements."""

    def save(self, path):
        self._to_npz(path, DATAFILE)
        np.savez(
            path / f"{SIMULATED_DENSITY_MATRIX}.npz",
            **{json.dumps(i): self.ideal[i] for i in self.ideal},
        )

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.data = super().load_data(path, DATAFILE)
        instance.ideal = super().load_data(path, SIMULATED_DENSITY_MATRIX)

        return instance


@dataclass
class StateTomographyResults(Results):
    """Tomography results"""

    measured_density_matrix_real: dict[QubitId, list]
    """Real part of measured density matrix."""
    measured_density_matrix_imag: dict[QubitId, list]
    """Imaginary part of measured density matrix."""
    target_density_matrix_real: dict[QubitId, list]
    """Real part of exact density matrix."""
    target_density_matrix_imag: dict[QubitId, list]
    """Imaginary part of exact density matrix."""
    fidelity: dict[QubitId, float]
    """State fidelity."""


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[QubitId]
) -> StateTomographyData:
    """Acquisition protocol for single qubit state tomography experiment."""

    if params.circuit is None:
        params.circuit = Circuit(len(targets))

    backend = GlobalBackend()
    transpiler = dummy_transpiler(backend)

    data = StateTomographyData()

    for basis in BASIS:
        basis_circuit = deepcopy(params.circuit)
        # FIXME: https://github.com/qiboteam/qibo/issues/1318
        if basis != "Z":
            for i in range(len(targets)):
                basis_circuit.add(getattr(gates, basis)(i).basis_rotation())

        basis_circuit.add(gates.M(i) for i in range(len(targets)))
        _, results = execute_transpiled_circuit(
            basis_circuit,
            targets,
            backend,
            nshots=params.nshots,
            transpiler=transpiler,
        )
        for i, target in enumerate(targets):
            data.register_qubit(
                TomographyType,
                (target, basis),
                dict(
                    samples=np.array(results.samples()).T[i],
                ),
            )
            data.ideal[target, basis] = np.array(
                NumpyBackend().execute_circuit(basis_circuit, nshots=10000).samples()
            ).T[i]
    return data


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""
    measured_density_matrix_real = {}
    measured_density_matrix_imag = {}
    target_density_matrix_real = {}
    target_density_matrix_imag = {}
    fid = {}
    for qubit in data.qubits:
        x_exp, y_exp, z_exp = (
            1 - 2 * np.mean(data[qubit, basis].samples) for basis in BASIS
        )
        density_matrix = 0.5 * (
            matrices.I + matrices.X * x_exp + matrices.Y * y_exp + matrices.Z * z_exp
        )
        measured_density_matrix_real[qubit] = np.real(density_matrix).tolist()
        measured_density_matrix_imag[qubit] = np.imag(density_matrix).tolist()

        x_theory, y_theory, z_theory = (
            1 - 2 * np.mean(data.ideal[qubit, basis]) for basis in BASIS
        )
        target_density_matrix = 0.5 * (
            matrices.I
            + matrices.X * x_theory
            + matrices.Y * y_theory
            + matrices.Z * z_theory
        )
        target_density_matrix_real[qubit] = np.real(target_density_matrix).tolist()
        target_density_matrix_imag[qubit] = np.imag(target_density_matrix).tolist()
        fid[qubit] = fidelity(
            np.array(measured_density_matrix_real[qubit])
            + 1.0j * np.array(measured_density_matrix_imag[qubit]),
            target_density_matrix,
        )

    return StateTomographyResults(
        measured_density_matrix_real=measured_density_matrix_real,
        measured_density_matrix_imag=measured_density_matrix_imag,
        target_density_matrix_real=target_density_matrix_real,
        target_density_matrix_imag=target_density_matrix_imag,
        fidelity=fid,
    )


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitId):
    """Plotting for state tomography"""
    fig = make_subplots(
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

    if fit is not None:
        # computing limits for colorscale
        min_re, max_re = np.min(fit.target_density_matrix_real[target]), np.max(
            fit.target_density_matrix_real[target]
        )
        min_im, max_im = np.min(fit.target_density_matrix_imag[target]), np.max(
            fit.target_density_matrix_imag[target]
        )

        # add offset
        if np.abs(min_re - max_re) < 1e-5:
            min_re = min_re - 0.1
            max_re = max_re + 0.1

        if np.abs(min_im - max_im) < 1e-5:
            min_im = min_im - 0.1
            max_im = max_im + 0.1
        fig.add_trace(
            go.Heatmap(
                z=fit.measured_density_matrix_real[target],
                x=["0", "1"],
                y=["0", "1"],
                colorscale="ice",
                colorbar_x=-0.2,
                zmin=min_re,
                zmax=max_re,
                reversescale=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.target_density_matrix_real[target],
                x=["0", "1"],
                y=["0", "1"],
                showscale=False,
                colorscale="ice",
                zmin=min_re,
                zmax=max_re,
                reversescale=True,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.measured_density_matrix_imag[target],
                x=["0", "1"],
                y=["0", "1"],
                colorscale="Burg",
                colorbar_x=1.01,
                zmin=min_im,
                zmax=max_im,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.target_density_matrix_imag[target],
                x=["0", "1"],
                y=["0", "1"],
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
