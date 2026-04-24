from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import (
    Parameter,
    PulseLike,
    PulseSequence,
    Sweeper,
)
from qibolab._core.components import IqChannel
from qibolab._core.identifier import ChannelId

from qibocal.auto.operation import Parameters, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from ..resonator_spectroscopies.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
    ResSpecType,
)

__all__ = [
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "QubitSpectroscopyData",
    "calculate_batches",
    "create_spectr_sweeper_and_updates",
    "QubitSpectrumType",
]

QubitSpectrumType = ResSpecType


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    hardware_average: bool = True
    """By default hardware average will be performed."""


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float] = field(default_factory=dict)
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Raw fitting output."""


@dataclass
class QubitSpectroscopyData(ResonatorSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""

    targets: list[QubitId] = field(default_factory=list)
    """List of qubits targeted in the experiment."""
    data: dict[QubitId, npt.NDArray[QubitSpectrumType]] = field(default_factory=dict)
    """Raw data acquired."""


def spectroscopy_sequence(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> tuple[
    PulseSequence,
    PulseSequence,
    dict[QubitId, PulseLike],
    dict[QubitId, ChannelId],
    dict[QubitId, ChannelId],
    dict[QubitId, float],
]:
    # Build the pulse sequence
    sequence = PulseSequence()
    ro_sequence = PulseSequence()
    ro_pulses = {}
    drive_channels = {}
    los_channels = {}
    drive_amplitudes = {}

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        ro_pulses[qubit] = ro_pulse
        drive_channels[qubit] = qd_channel
        drive_amplitudes[qubit] = qd_pulse.amplitude

        # Get the LO channel associated with this drive channel
        channel_obj = platform.channels[qd_channel]
        if isinstance(channel_obj, IqChannel):
            los_channels[qubit] = channel_obj.lo
        else:
            los_channels[qubit] = None

        sequence.append((qd_channel, qd_pulse))

        ro_sequence.append((ro_channel, ro_pulse))

    return (
        sequence,
        ro_sequence,
        ro_pulses,
        drive_channels,
        los_channels,
        drive_amplitudes,
    )


def calculate_batches(freq_width: int, max_if_bandwidth: int = 300_000_000):
    """
    Calculate frequency batches for wideband spectroscopy.

    """
    batch_starts = np.arange(-freq_width / 2, freq_width / 2, 2 * max_if_bandwidth)
    batch_ends = np.append(batch_starts[1:], freq_width / 2)
    batch_limits = np.stack((batch_starts, batch_ends))
    lo_offsets = batch_limits.sum(axis=0) / 2 if len(batch_starts) > 1 else [0]
    return np.vstack((batch_limits, lo_offsets)).T


def create_spectr_sweeper_and_updates(
    platform: CalibrationPlatform,
    targets: list[QubitId],
    drive_channels: dict[ChannelId],
    delta_frequency_range: npt.NDArray,
    los_channels: dict[ChannelId],
    lo_offset: float,
) -> tuple[dict[QubitId, Sweeper], dict[str, float]]:
    """Create a sweeper dictionary configuration for spectroscopy measurements on multiple qubits.

    This function creates a parallel sweeper that sweeps the frequency of drive channels
    across a specified range for each target qubit. It also prepares batch updates to
    synchronize the LO (local oscillator) frequencies if needed.
    The drive channel frequency is updated to f0 + lo_offset to avoid validation errors.
    LO channel frequencies are only updated if lo_offset is non-zero and the LO channel exists.
    """

    parsweep = {}
    batch_updates = {}
    for q in targets:
        f0 = platform.config(drive_channels[q]).frequency
        parsweep[q] = Sweeper(
            parameter=Parameter.frequency,
            values=f0 + delta_frequency_range,
            channels=[drive_channels[q]],
        )

        # Update the frequency of the drive channel to avoid raising a validation an error
        batch_updates[drive_channels[q]] = {"frequency": f0 + lo_offset}

        # If we're batching, update the LO
        if lo_offset != 0 and los_channels[q] is not None:
            batch_updates[los_channels[q]] = {"frequency": f0 + lo_offset}

    return parsweep, batch_updates
