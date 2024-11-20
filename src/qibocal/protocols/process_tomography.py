from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Parameters, Results, Routine

PREROTATIONS = [None, "x180", "y90", "-x90"]
POSTROTATIONS = [None, "-y90", "x90"]


ProcessTomographyType = np.dtype(
    [
        ("probabilities", float),
    ]
)
"""Custom dtype for tomography."""

Target = Union[QubitId, QubitPairId]
Moments = list[Union[tuple[str], tuple[str, str]]]


@dataclass
class ProcessTomographyParameters(Parameters):
    circuit: Moments = field(default_factory=list)
    """Gates to perform process tomography on."""

    def __post_init__(self):
        self.circuit = [tuple(moment) for moment in self.circuit]


@dataclass
class ProcessTomographyData(Data):
    """Tomography data."""

    prerotations: Moments
    circuit: Moments
    postrotations: Moments
    data: dict[Target, npt.NDArray[ProcessTomographyType]] = field(default_factory=dict)


def compile(
    moments: Moments, platform: Platform, qubits: list[QubitId]
) -> PulseSequence:
    sequence = PulseSequence()
    phases = defaultdict(float)
    for moment in moments:
        start = sequence.finish
        if moment[0] == "cz":
            cz_sequence, cz_phases = platform.pairs[
                tuple(qubits)
            ].native_gates.CZ.sequence(start=start)
            for q in qubits:
                phases[q] -= cz_phases[q]
            sequence += cz_sequence
        else:
            for q, gate in zip(qubits, moment):
                phase = phases[q]
                if gate == "x180":
                    sequence.add(
                        platform.create_RX_pulse(q, start=start, relative_phase=phase)
                    )
                elif gate == "y180":
                    sequence.add(
                        platform.create_RX_pulse(
                            q, start=start, relative_phase=np.pi / 2 + phase
                        )
                    )
                elif gate == "x90":
                    sequence.add(
                        platform.create_RX90_pulse(q, start=start, relative_phase=phase)
                    )
                elif gate == "y90":
                    sequence.add(
                        platform.create_RX90_pulse(
                            q, start=start, relative_phase=np.pi / 2 + phase
                        )
                    )
                elif gate == "-x90":
                    sequence.add(
                        platform.create_RX90_pulse(
                            q, start=start, relative_phase=np.pi + phase
                        )
                    )
                elif gate == "-y90":
                    sequence.add(
                        platform.create_RX90_pulse(
                            q, start=start, relative_phase=-np.pi / 2 + phase
                        )
                    )

    start = sequence.finish
    for q in qubits:
        sequence.add(platform.create_MZ_pulse(q, start=start))
    return sequence


def calculate_probabilities(samples: npt.NDArray) -> npt.NDArray:
    nshots, nqubits = samples.shape
    values, counts = np.unique(samples, axis=0, return_counts=True)
    freqs = {"".join([str(x) for x in v]): c for v, c in zip(values, counts)}
    assert sum(freqs.values()) == nshots
    outcomes = ["{:b}".format(x).zfill(nqubits) for x in range(2**nqubits)]
    return np.array([freqs[x] / nshots for x in outcomes])


def _acquisition(
    params: ProcessTomographyParameters, platform: Platform, targets: list[Target]
) -> ProcessTomographyData:
    """Acquisition protocol for two qubit state tomography experiment."""
    assert len(targets) == 1
    if not isinstance(targets[0], QubitId):
        qubits = list(targets[0])
    else:
        qubits = list(targets)

    prerotations = list(product(*[PREROTATIONS for _ in qubits]))
    postrotations = list(product(*[POSTROTATIONS for _ in qubits]))

    data = ProcessTomographyData(prerotations, params.circuit, postrotations)
    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
    )
    probabilities = []
    for prerot in prerotations:
        for postrot in postrotations:
            sequence = compile([prerot] + params.circuit + [postrot], platform, qubits)
            results = platform.execute_pulse_sequence(sequence, options)
            samples = np.stack([results[q].samples for q in qubits]).T
            probabilities.append(calculate_probabilities(samples))

    data.register_qubit(
        ProcessTomographyType,
        targets[0],
        {
            "probabilities": np.stack(probabilities),
        },
    )
    return data


@dataclass
class ProcessTomographyResults(Results):
    """Tomography results."""

    estimated_chi_real: dict[Target, list[list[float]]] = field(default_factory=dict)
    estimated_chi_imag: dict[Target, list[list[float]]] = field(default_factory=dict)
    target_chi_real: dict[Target, list[list[float]]] = field(default_factory=dict)
    target_chi_imag: dict[Target, list[list[float]]] = field(default_factory=dict)


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
        [to_circuit([rot], density_matrix=True)().state() for rot in rotations]
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
            to_circuit([rot] + data.circuit, density_matrix=True)().state()
            for rot in prerotations
        ]
        target_chi = calculate_chi(target_rhos, preparation_rotations=prerotations)

        results.estimated_chi_real[target] = estimated_chi.real.tolist()
        results.estimated_chi_imag[target] = estimated_chi.imag.tolist()
        results.target_chi_real[target] = target_chi.real.tolist()
        results.target_chi_imag[target] = target_chi.imag.tolist()

    return results


def plot_chi(estimated, target):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reconstruction", "Exact"))
    fig.add_trace(
        go.Heatmap(z=estimated, coloraxis="coloraxis"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=target, coloraxis="coloraxis"),
        row=1,
        col=2,
    )
    fig.update_layout(
        coloraxis=dict(
            colorscale="plasma",
        ),
    )
    # Flip the y-axes for both subplots
    fig.update_yaxes(
        autorange="reversed", row=1, col=1  # Flip the y-axis  # First subplot
    )
    fig.update_yaxes(
        autorange="reversed", row=1, col=2  # Flip the y-axis  # Second subplot
    )
    return fig


def _plot(data: ProcessTomographyData, fit: ProcessTomographyResults, target: Target):
    """Plotting for two qubit state tomography."""
    fitting_report = ""
    fig_real = plot_chi(fit.estimated_chi_real[target], fit.target_chi_real[target])
    fig_imag = plot_chi(fit.estimated_chi_imag[target], fit.target_chi_imag[target])
    fig_real.update_layout(title="Real")
    fig_imag.update_layout(title="Imag")
    return [fig_real, fig_imag], fitting_report


process_tomography = Routine(_acquisition, _fit, _plot)
