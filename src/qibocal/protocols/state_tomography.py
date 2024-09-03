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
    backend.platform = platform
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


def plot_parallelogram(a, e, pos_x, pos_y, **options):
    """Plotting single histogram in 3d plot."""
    x, y, z = np.meshgrid(
        np.linspace(pos_x - a / 4, pos_x + a / 4, 2),
        np.linspace(pos_y - a / 4, pos_y + a / 4, 2),
        np.linspace(0, e, 2),
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        alphahull=1,
        flatshading=True,
        lighting={"diffuse": 0.1, "specular": 2.0, "roughness": 0.5},
        **options,
    )


def plot_rho(fig, zz, trace_options, figure_options, showlegend=None):
    """Plot density matrix"""
    values = list(range(len(zz)))
    x, y = np.meshgrid(values, values)
    xx = x.flatten()
    yy = y.flatten()
    zz = np.array(zz).ravel()
    showlegend_temp = False
    for x, y, z in zip(xx, yy, zz):
        if showlegend is None:
            showlegend_temp = bool(x == xx[-1] and y == yy[-1])
        fig.add_trace(
            plot_parallelogram(1, z, x, y, showlegend=showlegend_temp, **trace_options),
            **figure_options,
        )


def plot_reconstruction(ideal, measured):
    """Plot 3D plot with reconstruction of ideal and measured density matrix."""
    fig = make_subplots(
        rows=1,
        cols=2,
        start_cell="top-left",
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "Re(ρ)",
            "Im(ρ)",
        ),
    )

    # computing limits for colorscale
    min_re, max_re = np.min(ideal.real), np.max(ideal.real)
    min_im, max_im = np.min(ideal.imag), np.max(ideal.imag)

    # add offset
    if np.abs(min_re - max_re) < 1e-5:
        min_re = min_re - 0.1
        max_re = max_re + 0.1
    if np.abs(min_im - max_im) < 1e-5:
        min_im = min_im - 0.1
        max_im = max_im + 0.1

    plot_rho(
        fig,
        measured.real,
        trace_options=dict(
            color="rgba(255,100,0,0.1)", name="experiment", legendgroup="experiment"
        ),
        figure_options=dict(row=1, col=1),
    )

    plot_rho(
        fig,
        ideal.real,
        trace_options=dict(
            color="rgba(100,0,100,0.1)", name="simulation", legendgroup="simulation"
        ),
        figure_options=dict(row=1, col=1),
    )

    plot_rho(
        fig,
        measured.imag,
        trace_options=dict(
            color="rgba(255,100,0,0.1)", name="experiment", legendgroup="experiment"
        ),
        figure_options=dict(row=1, col=2),
        showlegend=False,
    )

    plot_rho(
        fig,
        ideal.imag,
        trace_options=dict(
            color="rgba(100,0,100,0.1)", name="simulation", legendgroup="simulation"
        ),
        figure_options=dict(row=1, col=2),
        showlegend=False,
    )

    tickvals = list(range(len(ideal)))
    ticktext = ["{:02b}".format(i) for i in tickvals]
    fig.update_scenes(
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(tickvals=tickvals, ticktext=ticktext),
        zaxis=dict(range=[-1, 1]),
    )

    return fig


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitId):
    """Plotting for state tomography"""
    if fit is None:
        return [], ""

    ideal = np.array(fit.target_density_matrix_real[target]) + 1j * np.array(
        fit.target_density_matrix_imag[target]
    )
    measured = np.array(fit.measured_density_matrix_real[target]) + 1j * np.array(
        fit.measured_density_matrix_imag[target]
    )
    fig = plot_reconstruction(ideal, measured)

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


state_tomography = Routine(_acquisition, _fit, _plot)
