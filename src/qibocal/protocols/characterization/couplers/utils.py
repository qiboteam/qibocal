from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results

from ..flux_dependence.utils import create_data_array


@dataclass
class CouplerSpectroscopyParameters(Parameters):
    """CouplerResonatorSpectroscopy and CouplerQubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for frequency sweep [Hz]."""
    measured_qubits: list[QubitId]
    """Qubit to measure from the pair"""
    amplitude: Optional[float] = None
    """Readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    bias_width: Optional[float] = None
    """Width for bias sweep [V]."""
    bias_step: Optional[float] = None
    """Bias step for sweep [a.u.]."""
    flux_amplitude_start: Optional[Union[int, float, List[float]]] = None
    """Amplitude start value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]."""
    flux_amplitude_end: Optional[Union[int, float, List[float]]] = None
    """Amplitude end value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]."""
    flux_amplitude_step: Optional[Union[int, float, List[float]]] = None
    """Amplitude step(s) for flux pulses sweep [a.u.]."""

    def __post_init__(self):
        if not self.has_bias_params:
            if self.has_flux_params:
                self.check_flux_params()
                return
        if not self.has_flux_params:
            if self.has_bias_params:
                return
        raise ValueError(
            "Too many arguments provided. Provide either bias_width "
            "and bias_step or flux_amplitude_width and flux_amplitude_step."
        )

    def check_flux_params(self):
        """All flux params must be either all float or all lists with the same length.
        This function does not check if the lenght of the lists is equal to the number
        of qubits in the experiment.
        """
        flux_params = (
            self.flux_amplitude_start,
            self.flux_amplitude_end,
            self.flux_amplitude_step,
        )
        if all(isinstance(param, (int, float)) for param in flux_params):
            return

        if all(isinstance(param, list) for param in flux_params):
            if all(len(param) == len(flux_params[0]) for param in flux_params):
                return
            raise ValueError("Flux lists do not have the same length.")
        raise ValueError(
            "flux parameters have the wrong type. Expected one of (int, float, list)."
        )

    @property
    def has_bias_params(self):
        """True if both bias_width and bias_step are set."""
        return self.bias_width is not None and self.bias_step is not None

    @property
    def has_flux_params(self):
        """True if both biasflux_amplitude_width_width and flux_amplitude_step are set."""
        return (
            self.flux_amplitude_start is not None
            and self.flux_amplitude_end is not None
            and self.flux_amplitude_step is not None
        )

    @property
    def flux_pulses(self):
        """True if sweeping flux pulses, False if sweeping bias."""
        if self.has_flux_params:
            return True
        return False


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
    flux_pulses: bool
    """True if sweeping flux pulses, False if sweeping bias."""
    data: dict[QubitId, npt.NDArray[CouplerSpecType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = create_data_array(
            freq, bias, signal, phase, dtype=CouplerSpecType
        )

    @property
    def flux_pulses(self):
        """Return False because the experiment only supports bias."""
        return False
