"""Data structures and constants for randomized benchmarking."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from qibo import gates
from qibo.models import Circuit

from qibocal.auto.operation import Data, QubitId, QubitPairId, Results


class CircuitIndex(BaseModel):
    """Tracks the (depth, iteration) CircuitIndex of a circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int
    iteration: int


class IndexedCircuit(BaseModel):
    """A circuit paired with its (depth, iteration) CircuitIndex."""

    # arbitrary_types_allowed is needed to allow the Circuit type to be a field.
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    circuit: Circuit
    index: CircuitIndex


class IndexedResult(BaseModel):
    """An execution result paired with its (depth, iteration) CircuitIndex."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: Counter
    index: CircuitIndex


CircuitDepth = int


SINGLE_QUBIT_CLIFFORDS = {
    # Virtual gates
    0: gates.I,
    1: lambda q: gates.U3(q, 0, np.pi / 2, np.pi / 2),  # Z,
    2: lambda q: gates.U3(q, 0, np.pi / 2, 0),  # gates.RZ(q, np.pi / 2),
    3: lambda q: gates.U3(q, 0, -np.pi / 2, 0),  # gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.U3(q, np.pi, 0, np.pi),  # X,
    5: lambda q: gates.U3(q, np.pi, 0, 0),  # Y,
    # pi/2 rotations
    6: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),
    7: lambda q: gates.U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),
    8: lambda q: gates.U3(q, np.pi / 2, 0, 0),
    9: lambda q: gates.U3(q, -np.pi / 2, 0, 0),
    # 2pi/3 rotations
    10: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    11: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
    12: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
    13: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
    14: lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    15: lambda q: gates.U3(q, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
    16: lambda q: gates.U3(q, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
    17: lambda q: gates.U3(q, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    18: lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    19: lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    20: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
    21: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
    22: lambda q: gates.U3(q, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    23: lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}

NPULSES_PER_CLIFFORD = 1.875

"""
Global phases that could appear in the Clifford group we defined in the "2q_cliffords.json" file
due to the gates we selected to generate the Clifford group.
"""
GLOBAL_PHASES = [
    1 + 0j,
    -1 + 0j,
    0 + 1j,
    0 - 1j,
    0.707 + 0.707j,
    -0.707 + 0.707j,
    0.707 - 0.707j,
    -0.707 - 0.707j,
]


RBType = np.dtype(
    [
        ("survival_probs", np.float64),
    ]
)
"""Custom dtype for RB."""


@dataclass
class RBData(Data):
    """The output of the acquisition function."""

    depths: list[CircuitDepth]
    """Circuits depths."""
    uncertainties: float | None
    """Parameters uncertainties."""
    seed: int | None
    nshots: int
    """Number of shots."""
    niter: int
    """Number of iterations for each depth."""
    data: dict[
        tuple[QubitId, CircuitDepth] | tuple[QubitId, QubitId, CircuitDepth],
        npt.NDArray[RBType],
    ] = field(default_factory=dict)
    """Raw data acquired."""
    npulses_per_clifford: float = 1.875
    """Number of pulses for an average clifford."""


@dataclass
class RB2QData(RBData):
    """The output of the acquisition function."""

    npulses_per_clifford: float = 8.6  # Assuming U3s and 1 pulse CZ
    """Number of pulses for an average clifford."""

    def extract_probabilities(self, qubits):
        """Extract the probabilities given (`qubit`, `qubit`)"""
        probs = []
        for depth in self.depths:
            data_list = np.array(self.data[qubits[0], qubits[1], depth].tolist())
            data_list = data_list.reshape((-1, self.nshots))
            probs.append(np.count_nonzero(1 - data_list, axis=1) / data_list.shape[1])
        return probs


@dataclass
class RB2QInterData(RB2QData):
    """The output of the acquisition function."""

    fidelity: dict[QubitPairId, list] = field(default_factory=dict)
    """The interleaved fidelity of this qubit."""


@dataclass
class StandardRBResult(Results):
    """Standard RB outputs."""

    fidelity: dict[QubitId, float]
    """The overall fidelity of this qubit."""
    pulse_fidelity: dict[QubitId, float]
    """The pulse fidelity of the gates acting on this qubit."""
    fit_parameters: dict[QubitId, list[float]]
    """Raw fitting parameters."""
    fit_uncertainties: dict[QubitId, list[float]]
    """Fitting parameters uncertainties."""
    error_bars: dict[QubitId, float | list[float] | None] = field(default_factory=dict)
    """Error bars for y."""
