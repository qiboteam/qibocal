from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import Delay, Parameter, PulseSequence, Sweeper
from qibolab._core.components import IqChannel

from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase
from qibocal.update import replace

from ... import update
from ..resonator_spectroscopies.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
    ResSpecType,
)
from ..resonator_spectroscopies.resonator_utils import spectroscopy_plot
from ..utils import (
    chi2_reduced,
    lorentzian,
    lorentzian_fit,
)

__all__ = [
    "qubit_spectroscopy",
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "QubitSpectroscopyData",
    "_fit",
]


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

    frequency: dict[QubitId, dict[str, float]]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    chi2_reduced: dict[QubitId, tuple[float, Optional[float]]] = field(
        default_factory=dict
    )
    """Chi2 reduced."""
    error_fit_pars: dict[QubitId, list] = field(default_factory=dict)
    """Errors of the fit parameters."""


class QubitSpectroscopyData(ResonatorSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""


def _calculate_batches(freq_width: int, max_if_bandwidth: int = 300_000_000):
    """
    Calculate frequency batches for wideband spectroscopy.

    """
    batch_starts = np.arange(-freq_width / 2, freq_width / 2, 2 * max_if_bandwidth)
    batch_ends = np.append(batch_starts[1:], freq_width / 2)
    batch_limits = np.stack((batch_starts, batch_ends))
    lo_offsets = batch_limits.sum(axis=0) / 2 if len(batch_starts) > 1 else [0]
    return np.vstack((batch_limits, lo_offsets)).T


def _acquisition(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by batching when the frequency range exceeds ±300 MHz from the LO
    """

    # Calculate batches
    batches = _calculate_batches(params.freq_width)

    # Get drive channels and LO channels for each qubit
    drive_channels = {}
    lo_channels = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, _ = natives.RX()[0]
        drive_channels[qubit] = qd_channel

        # Get the LO channel associated with this drive channel
        channel_obj = platform.channels[qd_channel]
        if isinstance(channel_obj, IqChannel) and channel_obj.lo is not None:
            lo_channels[qubit] = channel_obj.lo
        else:
            lo_channels[qubit] = None

    # Initialize storage for intermediate results
    values = {qubit: defaultdict(list) for qubit in targets}
    amplitudes = {qubit: None for qubit in targets}

    # Execute each batch
    for start, end, lo_offset in batches:
        delta_frequency_range = np.arange(start, end, params.freq_step)

        # Build the pulse sequence
        sequence = PulseSequence()
        ro_pulses = {}
        sweepers = []

        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            qd_channel = drive_channels[qubit]
            _, qd_pulse = natives.RX()[0]
            ro_channel, ro_pulse = natives.MZ()[0]

            qd_pulse = replace(qd_pulse, duration=params.drive_duration)
            if params.drive_amplitude is not None:
                qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

            if qubit not in amplitudes:
                amplitudes[qubit] = qd_pulse.amplitude

            ro_pulses[qubit] = ro_pulse

            sequence.append((qd_channel, qd_pulse))
            sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((ro_channel, ro_pulse))

            f0 = platform.config(qd_channel).frequency
            sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=f0 + delta_frequency_range,
                    channels=[qd_channel],
                )
            )

        # Prepare updates for this batch
        batch_updates = []
        for qubit in targets:
            update_dict = {}

            # Update the frequency of the drive channel to avoid raising a validation an error
            update_dict[drive_channels[qubit]] = {
                "frequency": platform.config(drive_channels[qubit]).frequency
                + lo_offset
            }

            # If we're batching, update the LO
            if lo_offset != 0 and lo_channels[qubit] is not None:
                f0 = platform.config(drive_channels[qubit]).frequency
                update_dict[lo_channels[qubit]] = {"frequency": f0 + lo_offset}

            batch_updates.append(update_dict)

        # Execute this batch
        results = platform.execute(
            [sequence],
            [sweepers],
            updates=batch_updates,
            **params.execution_parameters,
        )

        # Collect results from this batch
        for qubit in targets:
            result = results[ro_pulses[qubit].id]
            f0 = platform.config(drive_channels[qubit]).frequency

            signal = magnitude(result)
            _phase = phase(result)

            if len(signal.shape) > 1:
                error_signal = np.std(signal, axis=0, ddof=1) / np.sqrt(signal.shape[0])
                signal = np.mean(signal, axis=0)
                error_phase = np.std(_phase, axis=0, ddof=1) / np.sqrt(_phase.shape[0])
                _phase = np.mean(_phase, axis=0)
            else:
                error_signal = None
                error_phase = None

            # Store results with absolute frequencies
            values[qubit]["frequency"].append(delta_frequency_range + f0)
            values[qubit]["signal"].append(signal)
            values[qubit]["phase"].append(_phase)
            values[qubit]["error_signal"].append(error_signal)
            values[qubit]["error_phase"].append(error_phase)

    # Create data structure and aggregate results
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type, amplitudes=amplitudes
    )

    # Combine all batches for each qubit
    for qubit in targets:
        # Concatenate arrays from all batches
        freq = np.concatenate(values[qubit]["frequency"])
        signal = np.concatenate(values[qubit]["signal"])
        _phase = np.concatenate(values[qubit]["phase"])

        # Handle when error signals are available
        if all(x is not None for x in values[qubit]["error_signal"]):
            error_signal = np.concatenate(values[qubit]["error_signal"])
            error_phase = np.concatenate(values[qubit]["error_phase"])
        else:
            error_signal = None
            error_phase = None

        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                signal=signal,
                phase=_phase,
                freq=freq,
                error_signal=error_signal,
                error_phase=error_phase,
            ),
        )

    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}
    error_fit_pars = {}
    chi2 = {}
    for qubit in qubits:
        fit_result = lorentzian_fit(
            data[qubit], resonator_type=data.resonator_type, fit="qubit"
        )
        if fit_result is not None:
            frequency[qubit], fitted_parameters[qubit], error_fit_pars[qubit] = (
                fit_result
            )
            chi2[qubit] = (
                chi2_reduced(
                    data[qubit].signal,
                    lorentzian(data[qubit].freq, *fitted_parameters[qubit]),
                    data[qubit].error_signal,
                ),
                np.sqrt(2 / len(data[qubit].freq)),
            )
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
        error_fit_pars=error_fit_pars,
        chi2_reduced=chi2,
    )


def _plot(data: QubitSpectroscopyData, target: QubitId, fit: QubitSpectroscopyResults):
    """Plotting function for QubitSpectroscopy."""
    return spectroscopy_plot(data, target, fit)


def _update(
    results: QubitSpectroscopyResults, platform: CalibrationPlatform, target: QubitId
):
    platform.calibration.single_qubits[target].qubit.frequency_01 = results.frequency[
        target
    ]
    update.drive_frequency(results.frequency[target], platform, target)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
