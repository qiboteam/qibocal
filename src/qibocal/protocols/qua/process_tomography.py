"""Process tomography based on https://arxiv.org/abs/quant-ph/9610001

Can be used to reconstruct the channel corresponding to the implementation
of a gate or sequence of gates on quantum hardware.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibolab import Platform

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)

from ..utils import table_dict, table_html
from .stream_circuits import execute

PREROTATIONS = [None, "x180", "y90", "-x90"]
POSTROTATIONS = [None, "-y90", "x90"]


ProcessTomographyType = np.dtype(
    [
        ("probabilities", float),
    ]
)
"""Custom dtype for process tomography."""

Target = Union[QubitId, QubitPairId]
"""Process tomography works on both single and two qubit circuits."""
Moments = list[Union[tuple[str], tuple[str, str]]]
"""Compact rerpresentation of a circuit on one or two qubits."""


@dataclass
class ProcessTomographyParameters(Parameters):
    circuit: Moments = field(default_factory=list)
    """Circuit for which we reconstruct the channel."""
    debug: Optional[str] = None

    def __post_init__(self):
        self.circuit = [tuple(moment) for moment in self.circuit]


@dataclass
class ProcessTomographyData(Data):
    """Tomography data."""

    prerotations: Moments
    """Gates used for state preparation."""
    circuit: Moments
    """Circuit for which we reconstruct the channel."""
    postrotations: Moments
    """Gates used for rotating to different basis before measurement."""
    data: dict[Target, npt.NDArray[ProcessTomographyType]] = field(default_factory=dict)
    """Measurement probabilities for all state preparations and measurement bases."""


def calculate_probabilities(samples: npt.NDArray) -> npt.NDArray:
    """Converts measurement samples to probabilities.

    Args:
        samples: Array of shape ``(nshots, nqubits)``.

        Returns:
            Array of probabilities of shape ``(2 ** nqubits,)``.
    """
    nshots, nqubits = samples.shape
    values, counts = np.unique(samples, return_counts=True, axis=0)
    freqs = {"".join([str(x) for x in v]): c for v, c in zip(values, counts)}
    assert sum(freqs.values()) == nshots
    outcomes = ["{:b}".format(x).zfill(nqubits) for x in range(2**nqubits)]
    return np.array([freqs.get(x, 0) / nshots for x in outcomes])


def _acquisition(
    params: ProcessTomographyParameters, platform: Platform, targets: list[Target]
) -> ProcessTomographyData:
    """Acquisition protocol for process tomography experiment on one or two qubits."""
    assert len(targets) == 1
    qubits = targets[0]
    assert len(qubits) == 2

    prerotations = list(product(*[PREROTATIONS for _ in qubits]))
    postrotations = list(product(*[POSTROTATIONS for _ in qubits]))

    data = ProcessTomographyData(prerotations, params.circuit, postrotations)

    sequences = []
    for prerot in prerotations:
        for postrot in postrotations:
            sequences.append([prerot] + params.circuit + [postrot])

    state0, state1 = execute(
        sequences, platform, qubits, params.nshots, params.relaxation_time, params.debug
    )
    shots = np.stack([state0, state1]).astype(int)

    from pathlib import Path

    np.save(Path.cwd() / "process_tomography_shots.npy", shots)

    probabilities = [
        calculate_probabilities(shots[:, i].T) for i in range(len(sequences))
    ]
    data.register_qubit(
        ProcessTomographyType,
        targets[0],
        {
            "probabilities": np.stack(probabilities),
        },
    )
    return data


GATE_MAP = {
    None: lambda q: gates.I(q),
    "x180": lambda q: gates.RX(q, theta=np.pi),
    "y180": lambda q: gates.RY(q, theta=np.pi),
    "x90": lambda q: gates.RX(q, theta=np.pi / 2),
    "y90": lambda q: gates.RY(q, theta=np.pi / 2),
    "-x90": lambda q: gates.RX(q, theta=-np.pi / 2),
    "-y90": lambda q: gates.RY(q, theta=-np.pi / 2),
}


def to_circuit(moments: Moments, density_matrix: bool = False) -> Circuit:
    nqubits = len(moments[0])
    circuit = Circuit(nqubits, density_matrix=density_matrix)
    for moment in moments:
        assert len(moment) == nqubits
        if moment[0] == "cz":
            circuit.add(gates.CZ(0, 1))
        else:
            for q, r in enumerate(moment):
                if r is not None:
                    circuit.add(GATE_MAP[r](q))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


def simulate_circuit(circuit: Circuit):
    backend = NumpyBackend()
    return backend.execute_circuit(circuit)


def basis_matrices(rotations: list[tuple[str]]) -> list[npt.NDArray]:
    matrices = []
    for rotation in rotations:
        g = rotation[0]
        matrices.append(GATE_MAP[g](0).matrix())
        for g in rotation[1:]:
            matrices[-1] = np.kron(matrices[-1], GATE_MAP[g](0).matrix())
    return matrices


def project_psd(matrix):
    """Project matrix to the space of positive semidefinite matrices."""
    s, v = np.linalg.eigh(matrix)
    s = s * (s > 0)
    return v.dot(np.diag(s)).dot(v.conj().T)


def state_tomography(data, rotations):
    matrices = basis_matrices(rotations)
    d = len(matrices[0])
    measurement = np.zeros((0, d**2))
    for u in matrices:
        channel = np.kron(u, u.conj())
        measure_channel = channel[np.eye(d, dtype=bool).flatten(), :]
        measurement = np.concatenate((measurement, measure_channel))

    rho_direct_estimate = np.linalg.pinv(measurement).dot(data.flatten())
    rho_direct_estimate = rho_direct_estimate.reshape((d, d))
    rho_direct_estimate_proj = project_psd(rho_direct_estimate)
    rho_direct_estimate_proj = rho_direct_estimate_proj / np.trace(
        rho_direct_estimate_proj
    )
    return rho_direct_estimate_proj


def calculate_ideal_basis(d: int):
    """Creates density matrix computational basis.

    Args:
        d: Density matrix dimension
    """
    basis = np.zeros([d**2, d, d], dtype=complex)
    for i in range(d):
        for j in range(d):
            basis[d * i + j, i, j] = 1
    return basis


def rotate_to_ideal(rhos, rotations):
    d = len(rotations)
    ideal_basis = calculate_ideal_basis(int(np.sqrt(d)))
    experiment_basis = np.array(
        [
            simulate_circuit(to_circuit([rot], density_matrix=True)).state()
            for rot in rotations
        ]
    )
    rotation = ideal_basis.reshape((d, d)).dot(
        np.linalg.inv(experiment_basis.reshape((d, d)))
    )
    return np.einsum("ij,jab->iab", rotation, rhos)


def calculate_beta(operators):
    d = len(operators)
    ideal_basis = calculate_ideal_basis(int(np.sqrt(d)))
    beta = np.empty(4 * (d,), dtype=complex)
    for m, am in enumerate(operators):
        for n, an in enumerate(operators):
            for i, rho in enumerate(ideal_basis):
                beta[m, n, i] = am.dot(rho.dot(an.conj().T)).flatten()
    return beta.reshape((d**2, d**2))


DEFAULT_OPERATORS = np.array(
    [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        -1j * np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
)


def default_operators(d: int):
    matrices = DEFAULT_OPERATORS
    if d == 4:
        return np.copy(matrices)
    elif d == 16:
        return np.array([np.kron(x, y) for x in matrices for y in matrices])
    raise NotImplementedError


def calculate_chi(rho_estimates, operators=None, preparation_rotations=None):
    """Calculate channel chi matrix using process tomography.

    Args:
        rho_estimates: Density matrix estimates (from state tomography) in
            the ideal basis.
        operators: Operator basis to write the channel on.
        preparation_rotations: Rotations used to prepare initial states.
            If not given, ideal basis is assumed.
    """
    d = len(rho_estimates)
    if operators is None:
        operators = default_operators(d)
    if preparation_rotations is not None:
        rho_estimates = rotate_to_ideal(rho_estimates, preparation_rotations)

    assert len(operators) == d
    beta = calculate_beta(operators)
    kappa = np.linalg.pinv(beta).T
    return kappa.dot(rho_estimates.flatten()).reshape((d, d))


@dataclass
class ProcessTomographyResults(Results):
    """Tomography results."""

    estimated_chi_real: dict[Target, list[list[float]]] = field(default_factory=dict)
    estimated_chi_imag: dict[Target, list[list[float]]] = field(default_factory=dict)
    target_chi_real: dict[Target, list[list[float]]] = field(default_factory=dict)
    target_chi_imag: dict[Target, list[list[float]]] = field(default_factory=dict)


def _fit(data: ProcessTomographyResults) -> ProcessTomographyResults:
    prerotations = data.prerotations
    postrotations = data.postrotations
    n = len(postrotations)
    results = ProcessTomographyResults()
    for target, values in data.data.items():
        probs = values["probabilities"]
        estimated_rhos = np.array(
            [
                state_tomography(probs[i * n : (i + 1) * n], postrotations)
                for i in range(len(prerotations))
            ]
        )
        estimated_chi = calculate_chi(
            estimated_rhos, preparation_rotations=prerotations
        )

        target_rhos = [
            simulate_circuit(
                to_circuit([rot] + data.circuit, density_matrix=True)
            ).state()
            for rot in prerotations
        ]
        target_chi = calculate_chi(target_rhos, preparation_rotations=prerotations)

        results.estimated_chi_real[target] = estimated_chi.real.tolist()
        results.estimated_chi_imag[target] = estimated_chi.imag.tolist()
        results.target_chi_real[target] = target_chi.real.tolist()
        results.target_chi_imag[target] = target_chi.imag.tolist()

    return results


def _plot(data: ProcessTomographyData, fit: ProcessTomographyResults, target: Target):
    """Plotting for two qubit state tomography."""
    fitting_report = table_html(
        table_dict(
            [target],
            ["Target circuit"],
            [str(data.circuit)],
        )
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Re(Reconstruction)",
            "Re(Exact)",
            "Im(Reconstruction)",
            "Im(Exact)",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Heatmap(z=fit.estimated_chi_real[target], coloraxis="coloraxis"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=fit.target_chi_real[target], coloraxis="coloraxis"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=fit.estimated_chi_imag[target], coloraxis="coloraxis"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=fit.target_chi_imag[target], coloraxis="coloraxis"),
        row=2,
        col=2,
    )
    fig.update_layout(
        height=800,
        coloraxis=dict(
            colorscale="RdBu",
        ),
    )
    # Flip the y-axes
    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_yaxes(autorange="reversed", row=row, col=col)
    return [fig], fitting_report


qua_process_tomography = Routine(_acquisition, _fit, _plot)
