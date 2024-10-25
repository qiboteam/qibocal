from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from qibolab._core.identifier import QubitId, QubitPairId
from qibolab._core.serialize import NdArray

CALIBRATION = "calibration.json"
"""Calibration file."""


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

    omega_01: Optional[float] = None
    """"0->1 transition frequency [Hz]."""
    omega_12: Optional[float] = None
    """1->2 transition frequency [Hz]."""
    asymmetry: Optional[float] = None
    """Junctions asymmetry."""
    sweetspot: Optional[float] = None
    """Qubit sweetspot [V]."""

    @property
    def anharmonicity(self):
        """Anharmonicity of the qubit [Hz]."""
        return self.omega_12 - self.omega_01

    @property
    def charging_energy(self):
        """Charging energy Ec [Hz]."""
        return -self.anharmonicity


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


class Coherence(Model):
    """Coherence times of qubit."""

    t1: Optional[int] = 0
    """Relaxation time [ns]."""
    t2: Optional[int] = 0
    """T2 of the qubit [ns]."""
    t2_spin_echo: Optional[int] = 0
    """T2 hanh echo [ns]."""


class QubitCalibration(Model):
    """Container for calibration of single qubit."""

    resonator: Resonator = Field(default_factory=Resonator)
    """Resonator calibration."""
    qubit: Qubit = Field(default_factory=Qubit)
    """Qubit calibration."""
    readout: Readout = Field(default_factory=Readout)
    """Readout information."""
    coherence: Coherence = Field(default_factory=Coherence)
    """Coherence times of the qubit."""
    rb_fidelity: Optional[float] = None
    """Standard rb pulse fidelity."""


class TwoQubitCalibration(Model):
    """Container for calibration of qubit pair."""

    rb_fidelity: Optional[float] = None
    """Two qubit standard rb fidelity."""
    cz_fidelity: Optional[float] = None
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
