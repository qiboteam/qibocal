from pathlib import Path
from typing import Annotated, Optional, Union

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer
from scipy.sparse import lil_matrix

from .serialize import NdArray, SparseArray, ndarray_serialize, sparse_serialize

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Qubit name."""

QubitPairId = Annotated[
    tuple[QubitId, QubitId],
    BeforeValidator(lambda p: tuple(p.split("-")) if isinstance(p, str) else p),
    PlainSerializer(lambda p: f"{p[0]}-{p[1]}"),
]
"""Qubit pair name."""

CALIBRATION = "calibration.json"
"""Calibration file."""

Measure = tuple[float, Optional[float]]
"""Measured is represented as two values: mean and error."""


class Model(BaseModel):
    """Global model, holding common configurations."""

    model_config = ConfigDict(extra="forbid")


class Resonator(Model):
    """Representation of resonator parameters."""

    bare_frequency: Optional[float] = None
    """Bare resonator frequency [Hz]."""
    dressed_frequency: Optional[float] = None
    """Dressed resonator frequency [Hz]."""
    depletion_time: Optional[int] = None
    """Depletion time [ns]."""
    bare_frequency_amplitude: Optional[float] = None
    """Readout amplitude at high frequency."""

    @property
    def dispersive_shift(self):
        """Dispersive shift."""
        return self.bare_frequency - self.dressed_frequency

    # TODO: Add setter for dispersive shift as well
    # TODO: Add something related to resonator calibration


class Qubit(Model):
    """Representation of Qubit parameters"""

    frequency_01: Optional[float] = None
    """"0->1 transition frequency [Hz]."""
    frequency_12: Optional[float] = None
    """1->2 transition frequency [Hz]."""
    maximum_frequency: Optional[float] = None
    """Maximum transition frequency [Hz]."""
    asymmetry: Optional[float] = None
    """Junctions asymmetry."""
    sweetspot: Optional[float] = None
    """Qubit sweetspot [V]."""
    flux_coefficients: Optional[list[float]] = None
    """Amplitude - frequency dispersion relation coefficients """

    @property
    def anharmonicity(self):
        """Anharmonicity of the qubit [Hz]."""
        if self.frequency_12 is None:
            return 0
        return self.frequency_12 - self.frequency_01

    @property
    def charging_energy(self):
        """Charging energy Ec [Hz]."""
        return -self.anharmonicity

    @property
    def josephson_energy(self):
        """Josephson energy [Hz].

        The following formula is the inversion of the maximum frequency
        obtained from the flux dependence protoco.

        """
        return (
            (self.maximum_frequency + self.charging_energy) ** 2
            / 8
            / self.charging_energy
        )


class Readout(Model):
    """Readout parameters."""

    fidelity: Optional[float] = None
    """Readout fidelity."""
    coupling: Optional[float] = None
    """Readout coupling [Hz]."""
    effective_temperature: Optional[float] = None
    """Qubit effective temperature."""
    ground_state: list[float] = Field(default_factory=list)
    """Ground state position in IQ plane."""
    excited_state: list[float] = Field(default_factory=list)
    """Excited state position in IQ plane."""
    qudits_frequency: dict[int, float] = Field(default_factory=dict)
    """Dictionary mapping state with readout frequency."""

    @property
    def assignment_fidelity(self):
        """Assignment fidelity."""
        return (1 + self.fidelity) / 2


class QubitCalibration(Model):
    """Container for calibration of single qubit."""

    resonator: Resonator = Field(default_factory=Resonator)
    """Resonator calibration."""
    qubit: Qubit = Field(default_factory=Qubit)
    """Qubit calibration."""
    readout: Readout = Field(default_factory=Readout)
    """Readout information."""
    t1: Optional[Measure] = None
    """Relaxation time [ns]."""
    t2: Optional[Measure] = None
    """T2 of the qubit [ns]."""
    t2_spin_echo: Optional[Measure] = None
    """T2 hanh echo [ns]."""
    rb_fidelity: Optional[Measure] = None
    """Standard rb pulse fidelity."""


class TwoQubitCalibration(Model):
    """Container for calibration of qubit pair."""

    rb_fidelity: Optional[Measure] = None
    """Two qubit standard rb fidelity."""
    cz_fidelity: Optional[Measure] = None
    """CZ interleaved rb fidelity."""
    coupling: Optional[float] = None
    """Qubit-qubit coupling."""


class Matrix(Model):
    """Generic matrix indexed by qubits."""

    qubits: list[QubitId] = Field(default_factory=list, exclude=True)
    """List of QubitIds."""
    matrix: Optional[NdArray] = None
    """Matrix array."""

    def __eq__(self, tmp):
        array_equal = np.testing.assert_array_equal(self.matrix, tmp.matrix) is None
        return self.qubits == tmp.qubits and array_equal

    def __getitem__(self, pair: tuple[QubitId, QubitId]) -> float:
        """Getting matrix element from QubitIds."""
        q0, q1 = pair
        return self.matrix[self.qubits.index(q0)][  # pylint: disable=E1136
            self.qubits.index(q1)
        ]

    def __setitem__(self, index, value) -> None:
        """Setting matrix element from QubitIds."""
        row, col = index
        self.matrix[self.qubits.index(row)][  # pylint: disable=E1136
            self.qubits.index(col)
        ] = value


class MitigationMatrix(Model):
    """Generic matrix indexed by qubits."""

    qubits: list[QubitId] = Field(default_factory=list, exclude=True)
    """List of QubitIds."""
    matrix: Optional[SparseArray] = None
    """Sparse matrix."""

    def __eq__(self, tmp):
        return self.qubits == tmp.qubits and (self.matrix != tmp.matrix).nnz == 0

    def __setitem__(self, targets: list[QubitId], matrix: Union[list, NdArray]) -> None:
        """Setting submatrix corresponding to targets."""
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        mask = sum(1 << self.qubits.index(i) for i in targets)
        nqubits = len(self.qubits)
        indices = [i for i in range(2**nqubits) if (i & mask) == i]
        self.matrix[np.ix_(indices, indices)] = matrix  # pylint: disable=E1137

    def __getitem__(self, targets: list[QubitId]) -> SparseArray:
        """Getting submatrix corresponding to targets."""
        mask = sum(1 << self.qubits.index(i) for i in targets)
        nqubits = len(self.qubits)
        indices = [i for i in range(2**nqubits) if (i & mask) == i]
        return self.matrix[np.ix_(indices, indices)]  # pylint: disable=E1136


class Calibration(Model):
    """Calibration container."""

    single_qubits: dict[QubitId, QubitCalibration] = Field(default_factory=dict)
    """Dict with single qubit calibration."""
    two_qubits: dict[QubitPairId, TwoQubitCalibration] = Field(default_factory=dict)
    """Dict with qubit pairs calibration."""
    readout_mitigation_matrix: Optional[MitigationMatrix] = None
    """Readout mitigation matrix."""
    flux_crosstalk_matrix: Optional[Matrix] = None
    """Crosstalk flux matrix."""
    drive_crosstalk_matrix: Optional[Matrix] = None
    """Drive crosstalk matrix."""

    def model_post_init(self, __context):
        serialized_zeros = ndarray_serialize(
            np.zeros((len(self.qubits), len(self.qubits)))
        )
        if self.flux_crosstalk_matrix is None:
            self.flux_crosstalk_matrix = Matrix(
                qubits=self.qubits, matrix=serialized_zeros
            )
        else:
            self.flux_crosstalk_matrix.qubits = self.qubits

        if self.drive_crosstalk_matrix is None:
            self.drive_crosstalk_matrix = Matrix(
                qubits=self.qubits, matrix=serialized_zeros
            )
        else:
            self.drive_crosstalk_matrix.qubits = self.qubits

        if self.readout_mitigation_matrix is None:
            serialized_sparse_zeros = sparse_serialize(
                lil_matrix((2**self.nqubits, 2**self.nqubits))
            )
            self.readout_mitigation_matrix = MitigationMatrix(
                qubits=self.qubits, matrix=serialized_sparse_zeros
            )
        else:
            self.readout_mitigation_matrix.qubits = self.qubits

    def dump(self, path: Path):
        """Dump calibration model."""
        (path / CALIBRATION).write_text(self.model_dump_json(indent=4))

    @property
    def qubits(self) -> list:
        """List of qubits available in the model."""
        return list(self.single_qubits)

    @property
    def nqubits(self) -> int:
        """Number of qubits available."""
        return len(self.qubits)
