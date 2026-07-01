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
    """Pulse duration [ns] range."""
    pulse_duration_start: float | None = None
    """Initial pulse duration [ns]."""
    pulse_duration_end: float | None = None
    """Final pulse duration [ns]."""
    pulse_duration_step: float | None = None
    """Step pulse duration [ns]."""
    pulse_amplitude: float | None = None
    """Pulse amplitude. Same for all qubits."""
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
class RabiAmplitudeParameters(Parameters):
    """RabiAmplitude experiments runcard inputs."""

    ampl_range: tuple[float, float, float] | None = None
    """Pulse minimum amplitude [a.u] range."""
    min_amp: float | None = None
    """Minimum pulse amplitude [a.u.]."""
    max_amp: float | None = None
    """Maximum pulse amplitude [a.u.]."""
    step_amp: float | None = None
    """Step pulse amplitude [a.u.]."""
    pulse_length: float | None = None
    """Pulse duration [ns]. Same for all qubits."""
    drive_lines: list[QubitId] | None = None
    """Drive lines to use for the qubits;
    must be the same length as the qubits list."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""

    @property
    def amplitude_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the duration times of the pulses.
        """
        if self.ampl_range is None:
            return (self.min_amp, self.max_amp, self.step_amp)
        return self.ampl_range

    def __post_init__(self):
        if any([d is None for d in self.amplitude_range]):
            raise InputError("Valid pulse amplitude range not inserted.")


@dataclass
class RabiLengthFrequencyParameters(RabiLengthParameters):
    """RabiChevronLength runcard inputs."""

    freq_range: tuple[int, int, int] | None = None
    """Frequency range as an offset."""
    min_freq: int | None = None
    """Minimum frequency as an offset."""
    max_freq: int | None = None
    """Maximum frequency as an offset."""
    step_freq: int | None = None
    """Frequency to use as step for the scan."""

    @property
    def frequency_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the duration times of the pulses.
        """
        if self.freq_range is None:
            return (self.min_freq, self.max_freq, self.step_freq)
        return self.freq_range

    def __post_init__(self):
        super().__post_init__()

        if any([f is None for f in self.frequency_range]):
            raise InputError("Valid frequency offset range not inserted.")


@dataclass
class RabiAmplitudeFrequencyParameters(RabiAmplitudeParameters):
    """RabiChevronAmplitude runcard inputs."""

    freq_range: tuple[int, int, int] | None = None
    """Frequency range as an offset."""
    min_freq: int | None = None
    """Minimum frequency as an offset."""
    max_freq: int | None = None
    """Maximum frequency as an offset."""
    step_freq: int | None = None
    """Frequency to use as step for the scan."""

    @property
    def frequency_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the duration times of the pulses.
        """
        if self.freq_range is None:
            return (self.min_freq, self.max_freq, self.step_freq)
        return self.freq_range

    def __post_init__(self):
        super().__post_init__()

        if any([f is None for f in self.frequency_range]):
            raise InputError("Valid frequency offset range not inserted.")


@dataclass
class RabiResults(Results):
    """Results container for outputs produced by Rabi protocols."""

    drive_lines: dict[QubitId, QubitId]
    """List of drive line used for each qubit."""
    length: dict[QubitId, int | list[float]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, float | list[float]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    rx90: bool
    """Pi or Pi_half calibration"""
    chi2: dict[QubitId, list[float]] = field(default_factory=dict)


@dataclass
class RabiFreqResults(RabiResults):
    """Results container for outputs produced by Rabi protocols."""

    frequency: dict[QubitId, float | list[float]] = field(default_factory=dict)
    """Drive frequency for each qubit."""


@dataclass
class RabiData(Data):
    """RabiLength experiments acquisition outputs."""

    rx90: bool
    """Pi or Pi_half calibration"""
    drive_lines: dict[QubitId, QubitId] = field(default_factory=dict)
    """List of drive line used for each qubit."""
    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse duration for each target qubit."""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse amplitude for each target qubit."""
    data: dict[QubitId, Any] = field(default_factory=dict)
    """Raw data acquired."""
