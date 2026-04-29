"""Process tomography based on https://arxiv.org/abs/quant-ph/9610001

Can be used to reconstruct the channel corresponding to the implementation
of a gate or sequence of gates on quantum hardware.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Union

import numpy as np
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibolab import Platform

from qibocal.auto.operation import Parameters, QubitId, QubitPairId, Routine
from qibocal.protocols.tomographies.two_qubit_state_tomography import (
    OUTCOMES,
    StateTomographyData,
    TomographyType,
    _fit,
    _plot,
)

from .stream_circuits import execute

BASIS = ["Z", "X", "Y"]
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
class StateTomographyParameters(Parameters):
    circuit: Moments = field(default_factory=list)
    """Circuit for which we reconstruct the channel."""

    def __post_init__(self):
        self.circuit = [tuple(moment) for moment in self.circuit]


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


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[Target]
) -> StateTomographyData:
    """Acquisition protocol for process tomography experiment on one or two qubits."""
    assert len(targets) == 1
    qubits = targets[0]
    assert len(qubits) == 2

    two_qubit_basis = list(product(*[BASIS for _ in qubits]))
    postrotations = list(product(*[POSTROTATIONS for _ in qubits]))

    simulated_state = simulate_circuit(to_circuit(params.circuit))
    simulated_dm = simulate_circuit(to_circuit(params.circuit, density_matrix=True))
    data = StateTomographyData(
        ideal={qubits: simulated_dm.state()}, simulated=simulated_state
    )

    sequences = [params.circuit + [postrot] for postrot in postrotations]

    state0, state1 = execute(
        sequences, platform, qubits, params.nshots, params.relaxation_time
    )
    shots = np.stack([state0, state1]).astype(int)
    for i, (basis1, basis2) in enumerate(two_qubit_basis):
        simulation_result = simulate_circuit(to_circuit(sequences[i]))
        simulation_probabilities = simulation_result.probabilities()

        values, counts = np.unique(shots[:, i].T, return_counts=True, axis=0)
        frequencies = {"".join([str(x) for x in v]): c for v, c in zip(values, counts)}

        data.register_qubit(
            TomographyType,
            tuple(qubits) + (basis1, basis2),
            {
                "frequencies": np.array([frequencies[i] for i in OUTCOMES]),
                "simulation_probabilities": simulation_probabilities,
            },
        )
    return data


qua_state_tomography = Routine(_acquisition, _fit, _plot)
