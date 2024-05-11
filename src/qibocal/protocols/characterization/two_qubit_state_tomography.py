import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from itertools import product
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import Circuit, gates
from qibo.backends import GlobalBackend, NumpyBackend, matrices
from qibo.quantum_info import fidelity
from qibo.result import QuantumState
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import DATAFILE, Data, Parameters, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit

from .utils import table_dict, table_html


PAULI_BASIS = ["X", "Y", "Z"]
OUTCOMES = ["00", "01", "10", "11"]

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
        ("frequencies", np.int64),
        ("simulation_probabilities", np.float64)
    ]
)
"""Custom dtype for tomography."""


@dataclass
class StateTomographyData(Data):
    """Tomography data"""

    data: dict[tuple[QubitPairId, tuple[str, str]], TomographyType] = field(default_factory=dict)

    def save(self, path):
        self._to_npz(path, DATAFILE)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.data = super().load_data(path, DATAFILE)
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
    params: StateTomographyParameters, platform: Platform, targets: list[QubitPairId]
) -> StateTomographyData:
    """Acquisition protocol for single qubit state tomography experiment."""
    qubits = [q for pair in targets for q in pair]
    for q, counts in Counter(qubits).items():
        if counts > 1:
            raise ValueError(f"Qubits can only be measured once, but qubit {q} is measured {counts} times.")

    if params.circuit is None:
        params.circuit = Circuit(len(qubits))

    backend = GlobalBackend()
    transpiler = dummy_transpiler(backend)

    data = StateTomographyData()

    for basis1, basis2 in product(PAULI_BASIS, PAULI_BASIS):
        basis_circuit = deepcopy(params.circuit)
        # FIXME: basis
        if basis1 != "Z":
            basis_circuit.add(getattr(gates, basis1)(2 * i).basis_rotation() for i in range(len(targets)))
        if basis2 != "Z":
            basis_circuit.add(getattr(gates, basis2)(2 * i + 1).basis_rotation() for i in range(len(targets)))

        basis_circuit.add(gates.M(2 * i, 2 * i + 1, register_name=f"reg{i}") for i in range(len(targets)))

        simulation_result = NumpyBackend().execute_circuit(basis_circuit)
        _, results = execute_transpiled_circuit(
            basis_circuit,
            qubits,
            backend,
            nshots=params.nshots,
            transpiler=transpiler,
        )
        
        for i, target in enumerate(targets):
            frequencies = results.frequencies(registers=True)[f"reg{i}"]
            simulation_probabilities = simulation_result.probabilities(qubits=(2 * i, 2 * i + 1))
            data.register_qubit(
                TomographyType,
                (target, (basis1, basis2)),
                {
                    "frequencies": np.array([frequencies[i] for i in OUTCOMES]),
                    "simulation_probabilities": simulation_probabilities,
                },
            )
        # TODO: Save state here
        #setattr(
        #    data,
        #    f"{basis1.lower()}{basis2.lower()}_basis_state",
        #    simulation_result
        #)
    return data


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""
    return StateTomographyResults(
        measured_density_matrix_real=0,
        measured_density_matrix_imag=0,
        target_density_matrix_real=0,
        target_density_matrix_imag=0,
        fidelity=1,
    )


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitPairId):
    """Plotting for state tomography"""
    fig = make_subplots(
        rows=3,
        cols=3,
        # "$\text{Plot 1}$"
        subplot_titles=tuple(
            f"{basis1}<sub>1</sub>{basis2}<sub>1</sub>"
            for basis1, basis2 in product(PAULI_BASIS, PAULI_BASIS)
        ),
    )

    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"
    for i, (basis1, basis2) in enumerate(product(PAULI_BASIS, PAULI_BASIS)):
        row = i // 3 + 1
        col = i % 3 + 1
        basis_data = data.data[target, (basis1, basis2)]

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
    fig.update_layout(barmode="overlay")#, height=900)
    
    return [fig], ""


two_qubit_state_tomography = Routine(_acquisition, _fit, _plot)
