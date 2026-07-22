from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)
from scipy.sparse import lil_matrix

from .serialize import NdArray, SparseArray, eq

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

    bare_frequency: float | None = None
    """Bare resonator frequency [Hz]."""
    dressed_frequency: float | None = None
    """Dressed resonator frequency [Hz]."""
    depletion_time: int | None = None
    """Depletion time [ns]."""
    bare_frequency_amplitude: float | None = None
    """Readout amplitude at high frequency."""

    @property
    def dispersive_shift(self):
        """Dispersive shift."""
        return self.bare_frequency - self.dressed_frequency

    # TODO: Add setter for dispersive shift as well
    # TODO: Add something related to resonator calibration


class Qubit(Model):
    """Representation of Qubit parameters"""

    frequency_01: float | None = None
    """"0->1 transition frequency [Hz]."""
    frequency_12: float | None = None
    """1->2 transition frequency [Hz]."""
    maximum_frequency: float | None = None
    """Maximum transition frequency [Hz]."""
    asymmetry: float | None = None
    """Junctions asymmetry."""
    sweetspot: float | None = None
    """Qubit sweetspot [V]."""
    flux_coefficients: list[float] | None = None
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

    def detuning(self, amplitude):
        if self.flux_coefficients is None:
            return 0
        return np.polyval(self.flux_coefficients, amplitude)


class Readout(Model):
    """Readout parameters."""

    fidelity: float | None = None
    """Readout fidelity."""
    coupling: float | None = None
    """Readout coupling [Hz]."""
    effective_temperature: float | None = None
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
    t1: Measure | None = None
    """Relaxation time [ns]."""
    t2: Measure | None = None
    """T2 of the qubit [ns]."""
    t2_spin_echo: Measure | None = None
    """T2 hanh echo [ns]."""
    rb_fidelity: Measure | None = None
    """Standard rb pulse fidelity."""
    rabi_ampl_oscillation: dict[Literal["signal", "classification"], float] = Field(
        default_factory=dict
    )
    """Calibrated rabi oscillations used as reference for mw crosstalk."""


class TwoQubitCalibration(Model):
    """Container for calibration of qubit pair."""

    rb_fidelity: Measure | None = None
    """Two qubit standard rb fidelity."""
    cz_fidelity: Measure | None = None
    """CZ interleaved rb fidelity."""
    coupling: list[float] | None = None
    """Qubit-qubit coupling."""


class Calibration(Model):
    """Calibration container."""

    single_qubits: dict[QubitId, QubitCalibration] = Field(default_factory=dict)
    """Dict with single qubit calibration."""
    two_qubits: dict[QubitPairId, TwoQubitCalibration] = Field(default_factory=dict)
    """Dict with qubit pairs calibration."""
    readout_mitigation_matrix: SparseArray | None = None
    """Readout mitigation matrix."""
    flux_crosstalk_matrix: NdArray | None = None
    """Crosstalk flux matrix."""
    microwave_crosstalk_matrix: NdArray | None = None
    """Microwave crosstalk matrix (complex)."""

    @model_validator(mode="after")
    def validate_flux_matrix(self):
        """Validate or initialize the flux crosstalk matrix.

        If the matrix is missing or empty, initialize it as an identity matrix of
        shape (nqubits, nqubits). If provided, verify the matrix shape is correct.
        """
        if self.flux_crosstalk_matrix is None or self.flux_crosstalk_matrix.size == 0:
            self.flux_crosstalk_matrix = np.eye(self.nqubits)
        elif self.flux_crosstalk_matrix.shape != (self.nqubits, self.nqubits):
            raise ValueError(
                "Drive crosstalk matrix must have shape (nqubits, nqubits)."
            )
        return self

    @model_validator(mode="after")
    def validate_microwave_matrix(self):
        """Validate or initialize the microwave crosstalk matrix.

        If the matrix is missing or empty, initialize it as a complex-valued
        matrix of shape (nqubits, nqubits) filled with infinity. If provided,
        verify the matrix shape is correct and convert its dtype to complex.
        """
        if (
            self.microwave_crosstalk_matrix is None
            or self.microwave_crosstalk_matrix.size == 0
        ):
            self.microwave_crosstalk_matrix = np.full(
                (self.nqubits, self.nqubits), np.inf, dtype=complex
            )
        elif self.microwave_crosstalk_matrix.shape != (self.nqubits, self.nqubits):
            raise ValueError(
                "Microwave crosstalk matrix must have shape (nqubits, nqubits)."
            )
        else:
            self.microwave_crosstalk_matrix.astype(complex)
        return self

    @property
    def qubits(self) -> list:
        """List of qubits available in the model."""
        return list(self.single_qubits)

    @property
    def nqubits(self) -> int:
        """Number of qubits available."""
        return len(self.qubits)

    def __eq__(self, other: "Calibration") -> bool:
        """Compare calibrations.

        The comparison requires adaption, since it may involve NumPy arrays, which do
        not generate a single boolean as output of the comparison operator.
        """
        return eq(self, other)

    def dump(self, path: Path):
        """Dump calibration model."""
        (path / CALIBRATION).write_text(
            self.model_dump_json(indent=4), encoding="utf-8"
        )

    def _readout_mitigation_matrix_indices(self, target: tuple[QubitId, ...]):
        mask = sum(1 << self.qubits.index(i) for i in target)
        indices = [i for i in range(2**self.nqubits) if (i & mask) == i]
        return np.ix_(indices, indices)

    def set_readout_mitigation_matrix_element(
        self,
        target: list[QubitId],
        readout_mitigation_dict: dict[tuple[QubitId, ...], npt.NDArray[np.float64]],
    ):
        # create empty matrix if it doesn't exist
        if self.readout_mitigation_matrix is None:
            self.readout_mitigation_matrix = lil_matrix(
                (2**self.nqubits, 2**self.nqubits)
            )
        # compute indices
        ids = self._readout_mitigation_matrix_indices(target)
        # update matrix
        self.readout_mitigation_matrix[ids] = readout_mitigation_dict[tuple(target)]

    def get_readout_mitigation_matrix_element(
        self, target: list[QubitId]
    ) -> SparseArray:
        assert self.readout_mitigation_matrix is not None
        ids = self._readout_mitigation_matrix_indices(target)
        return self.readout_mitigation_matrix[ids]

    def _get_element(
        self, matrix: NdArray, qubit: QubitId, line: QubitId
    ) -> float | complex:
        """Return the matrix element for a target qubit and a physical line."""
        return matrix[self.qubits.index(qubit), self.qubits.index(line)]

    def get_flux_crosstalk(self, qubit: QubitId, flux_line: QubitId) -> float:
        """Return the flux crosstalk coefficient for a target qubit and a flux line."""
        return self._get_element(
            matrix=self.flux_crosstalk_matrix,
            qubit=qubit,
            line=flux_line,
        )

    def get_microwave_crosstalk(
        self, qubit: QubitId, microwave_line: QubitId
    ) -> tuple[float, float]:
        """Return the microwave crosstalk coefficient (module and phase) for a target qubit and a microwave line."""
        complex_el = self._get_element(
            matrix=self.microwave_crosstalk_matrix,
            qubit=qubit,
            line=microwave_line,
        )
        return np.abs(complex_el), np.angle(complex_el)

    def _set_element(
        self, matrix: NdArray, qubit: QubitId, line: QubitId, value: float | complex
    ) -> NdArray:
        """Set the matrix element for a target qubit and a physical line."""
        matrix[self.qubits.index(qubit), self.qubits.index(line)] = value
        return matrix

    def set_flux_crosstalk(
        self, qubit: QubitId, flux_line: QubitId, value: float
    ) -> None:
        """Set the flux crosstalk coefficient for a target qubit and a flux line."""
        self.flux_crosstalk_matrix = self._set_element(
            matrix=self.flux_crosstalk_matrix,
            qubit=qubit,
            line=flux_line,
            value=value,
        )

    def set_microwave_crosstalk(
        self, qubit: QubitId, microwave_line: QubitId, module: float, phase: float = 0
    ) -> None:
        """Return the microwave crosstalk coefficient (module and phase) for a target qubit and a microwave line."""
        if np.isfinite(module):
            complex_val = module * np.exp(1j * phase)
        else:
            complex_val = np.inf + 0j
        self.microwave_crosstalk_matrix = self._set_element(
            matrix=self.microwave_crosstalk_matrix,
            qubit=qubit,
            line=microwave_line,
            value=complex_val,
        )
