from dataclasses import dataclass, field
from typing import Any

from qibocal.auto.operation import Data, Parameters, QubitId, Results


class InputError(Exception):
    """Raised when Rabi length protocol input validation fails."""

    pass


@dataclass
class RabiLengthParameters(Parameters):
    """RabiLength experiments runcard inputs."""

    pulse_duration_range: tuple[float, float, float] | None = None
    """Pi pulse duration [ns] range."""
    pulse_duration_start: float | None = None
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float | None = None
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float | None = None
    """Step pi pulse duration [ns]."""
    pulse_amplitude: float | None = None
    """Pi pulse amplitude. Same for all qubits."""
    drive_lines: list[QubitId] | None = None
    """Drive lines to use for the qubits;
    must be the same length as the qubits list."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""

    @property
    def duration_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the duration times of the pulses.
        """
        if self.pulse_duration_range is None:
            return (
                self.pulse_duration_start,
                self.pulse_duration_end,
                self.pulse_duration_step,
            )
        return self.pulse_duration_range

    def __post_init__(self):
        if any([d is None for d in self.duration_range]):
            raise InputError("Valid pulse duration range not inserted.")


@dataclass
class RabiLengthResults(Results):
    """Results container for outputs produced by RabiLength protocols."""

    length: dict[QubitId, int | list[float]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, float | list[float]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    rx90: bool
    """Pi or Pi_half calibration"""


@dataclass
class RabiLengthData(Data):
    """RabiLength experiments acquisition outputs."""

    rx90: bool
    """Pi or Pi_half calibration"""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, Any] = field(default_factory=dict)
    """Raw data acquired."""
