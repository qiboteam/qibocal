from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Results

from ..flux_dependence.resonator_flux_dependence import ResonatorFluxParameters
from ..flux_dependence.utils import create_data_array


@dataclass
class CouplerSpectroscopyParameters(ResonatorFluxParameters):
    """CouplerResonatorSpectroscopy and CouplerQubitSpectroscopy runcard inputs."""

    measured_qubits: Optional[list[QubitId]] = None
    """Qubit to measure from the pair"""
    amplitude: Optional[Union[int, float]] = None
    """Readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""


CouplerSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
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
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Qubit bias offset."""
    data: dict[QubitId, npt.NDArray[CouplerSpecType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = create_data_array(
            freq, bias, signal, phase, dtype=CouplerSpecType
        )
