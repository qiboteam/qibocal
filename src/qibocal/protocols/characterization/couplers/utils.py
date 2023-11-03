from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results

from ..flux_dependence.utils import create_data_array


@dataclass
class CouplerSpectroscopyParameters(Parameters):
    """CouplerResonatorSpectroscopy and CouplerQubitSpectroscopy runcard inputs."""

    bias_width: int
    """Width for bias (V)."""
    bias_step: int
    """Frequency step for bias sweep (V)."""
    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for frequency sweep (Hz)."""
    # TODO: It may be better not to use readout multiplex to avoid readout crosstalk
    measured_qubits: list[QubitId]
    """Qubit to readout from the pair"""
    amplitude: Optional[float] = None
    """Readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


CouplerSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for coupler resonator spectroscopy."""


@dataclass
class CouplerSpectroscopyResults(Results):
    """CouplerResonatorSpectroscopy or CouplerQubitSpectroscopy outputs."""

    sweetspot: dict[QubitId, float]
    """Sweetspot for each coupler."""
    pulse_amp: dict[QubitId, float]
    """Pulse amplitude for the coupler."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


@dataclass
class CouplerSpectroscopyData(Data):
    """Data structure for CouplerResonatorSpectroscopy or CouplerQubitSpectroscopy."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[CouplerSpecType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        self.data[qubit] = create_data_array(
            freq, bias, msr, phase, dtype=CouplerSpecType
        )
