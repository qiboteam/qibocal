from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer
from qibolab._core.identifier import QubitId, QubitPairId
from qibolab._core.serialize import NdArray

CALIBRATION = "calibration.json"
"""Calibration file."""

# TODO: convert to int if used only for coherence values
Measure = Annotated[
    tuple[float, Optional[float]],
    BeforeValidator(lambda p: tuple(p.split("+/-")) if isinstance(p, str) else p),
    PlainSerializer(lambda p: f"{p[0]}+/-{p[1]}"),
]
"""Measure serialized in runcard."""


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

    @property
    def anharmonicity(self):
        """Anharmonicity of the qubit [Hz]."""
        return self.frequency_12 - self.frequency_01

    @property
    def charging_energy(self):
        """Charging energy Ec [Hz]."""
        return -self.anharmonicity

    @property
    def josephson_energy(self):
        """Josephson energy [Hz]."""
        # TODO: Add josephson energy


class Readout(Model):
    """Readout parameters."""

    fidelity: Optional[float] = None
    """Readout fidelity."""
    effective_temperature: Optional[float] = None
    """Qubit effective temperature."""
    ground_state: list[float] = Field(default_factory=list)
    """Ground state position in IQ plane."""
    excited_state: list[float] = Field(default_factory=list)
    """Excited state position in IQ plane."""

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


class Calibration(Model):
    """Calibration container."""

    single_qubits: dict[QubitId, QubitCalibration] = Field(default_factory=dict)
    """Dict with single qubit calibration."""
    two_qubits: dict[QubitPairId, TwoQubitCalibration] = Field(default_factory=dict)
    """Dict with qubit pairs calibration."""
    readout_mitigation_matrix: Optional[NdArray] = None
    """Readout mitigation matrix."""
    flux_crosstalk_matrix: Optional[NdArray] = None
    """Crosstalk flux matrix."""

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

    # TODO: add crosstalk object where I can do this
    def get_crosstalk_element(self, qubit1: QubitId, qubit2: QubitId):
        a, b = self.qubits.index(qubit1), self.qubits.index(qubit2)
        return self.flux_crosstalk_matrix[a, b]

    def set_crosstalk_element(self, qubit1: QubitId, qubit2: QubitId, value: float):
        a, b = self.qubits.index(qubit1), self.qubits.index(qubit2)
        self.flux_crosstalk_matrix[a, b] = value
