from pydantic import BaseModel, ConfigDict, Field
from qibolab._core.identifier import QubitId, QubitPairId
from qibolab._core.serialize import NdArray


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


class Readout(Model):
    """Readout parameters."""

    assignment_fidelity: float = 0
    """Assignment fidelity."""
    readout_fidelity: float = 0
    """Readout fidelity."""
    ground_state: list[float] = Field(default_factory=list)
    """Ground state position in IQ plane."""
    excited_state: list[float] = Field(default_factory=list)
    """Excited state position in IQ plane."""


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

    resonator: Resonator
    """Resonator calibration."""
    qubit: Qubit
    """Qubit calibration."""
    readout: Readout
    """Readout information."""
    coherence: Coherence
    """Coherence times of the qubit."""
    rb_fidelity: float
    """Standard rb pulse fidelity."""


class TwoQubitCalibration(Model):
    """Container for calibration of qubit pair."""

    gate_fidelity: float = 0
    """Two qubit standard rb fidelity."""
    cz_fidelity: float = 0
    """CZ interleaved rb fidelity."""


class Calibration(Model):
    """Calibration container."""

    single_qubits: dict[QubitId, QubitCalibration] = Field(default_factory=dict)
    """Dict with single qubit calibration."""
    two_qubits: dict[QubitPairId, TwoQubitCalibration] = Field(default_factory=dict)
    """Dict with qubit pairs calibration."""
    readout_mitigation_matrix: NdArray = None
    """Readout mitigation matrix."""
    flux_crosstalk_matrix: NdArray = None
    """Crosstalk flux matrix."""
