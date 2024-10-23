from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# from qibolab._core.identifier import QubitId, QubitPairId
# from qibolab._core.serialize import NdArray

CALIBRATION = "calibration.json"
"""Calibration file."""


class Model(BaseModel):
    """Global qibolab model, holding common configurations."""

    model_config = ConfigDict(extra="forbid")


class Resonator(Model):
    """Representation of resonator parameters."""

    bare_frequency: float = 0
    """Bare resonator frequency [Hz]."""
    dressed_frequency: float = 0
    """Dressed resonator frequency [Hz]."""
    depletion_time: int = 0
    """Depletion time [ns]."""

    # TODO: Add something related to resonator calibration


class Qubit(Model):
    """Representation of Qubit parameters"""

    omega_01: float = 0
    """"0->1 transition frequency."""
    omega_12: float = 0
    """1->2 transition frequency."""
    asymmetry: float = 0
    """Junctions asymmetry."""
    sweetspot: float = 0
    """Qubit sweetspot [V]."""

    @property
    def anharmonicity(self):
        """Anharmonicity of the qubit in Hz."""
        return self.omega_12 - self.omega_01

    @property
    def charging_energy(self):
        """Charging energy Ec."""
        return -self.anharmonicity


class Readout(Model):
    """Readout parameters."""

    fidelity: float = 0
    """Readout fidelity."""
    effective_temperature: float = 0
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

    t1: int = 0
    """Relaxation time [ns]."""
    t2: int = 0
    """T2 of the qubit [ns]."""
    t2_spin_echo: int = 0
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
    rb_fidelity: float = 0
    """Standard rb pulse fidelity."""


class TwoQubitCalibration(Model):
    """Container for calibration of qubit pair."""

    rb_fidelity: float = 0
    """Two qubit standard rb fidelity."""
    cz_fidelity: float = 0
    """CZ interleaved rb fidelity."""


class Calibration(Model):
    """Calibration container."""

    single_qubits: dict[str, QubitCalibration] = Field(default_factory=dict)
    """Dict with single qubit calibration."""
    # TODO: dump pair as str instead of tuple
    two_qubits: dict[tuple, TwoQubitCalibration] = Field(default_factory=dict)
    """Dict with qubit pairs calibration."""
    # TODO: fix this as well
    readout_mitigation_matrix: str = None
    """Readout mitigation matrix."""
    flux_crosstalk_matrix: str = None
    """Crosstalk flux matrix."""

    def dump(self, path: Path):
        """Dump platform."""
        (path / CALIBRATION).write_text(self.model_dump_json(indent=4))
