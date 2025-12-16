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
    readout_frequency,
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


def _calculate_batches(freq_width: int, freq_step: int, max_if_bandwidth: int = 300_000_000):
    """
    Calculate frequency batches for wideband spectroscopy.
    
    When the requested frequency sweep width exceeds the IF bandwidth range (+/- max_if_bandwidth)
    we split it into multiple batches.
    
    Parameters:
    -----------
    freq_width : int
        Total frequency width to sweep [Hz]
    freq_step : int
        Frequency step [Hz]
    max_if_bandwidth : int
        Maximum IF bandwidth [Hz] from LO, default 300 MHz
        
    Returns:
    --------
    list of dict
        Each dict contains:
        - 'delta_freq_range': frequency offsets from center for this batch
        - 'lo_offset': LO frequency offset from original drive frequency
    """
    # If freq_width fits within the IF bandwidth, no batching needed
    if freq_width <= 2 * max_if_bandwidth:
        delta_frequency_range = np.arange(
            -freq_width / 2, freq_width / 2, freq_step
        )
        return [{'delta_freq_range': delta_frequency_range, 'lo_offset': 0}]
    
    # Calculate number of batches needed
    # Each batch covers 2 * max_if_bandwidth
    batch_width = 2 * max_if_bandwidth
    num_batches = int(np.ceil(freq_width / batch_width))
    
    # Calculate starting frequency (relative to center)
    start_freq = -freq_width / 2
    
    batches = []
    for batch_idx in range(num_batches):
        # Calculate this batch's frequency range
        batch_start = start_freq + batch_idx * batch_width
        batch_end = min(batch_start + batch_width, freq_width / 2)
        
        # Center of this batch
        batch_center = (batch_start + batch_end) / 2
        
        # LO should set to the batch center (maybe better to the batch edge and make smaller sweeps?)
        lo_offset = batch_center
        
        # Frequency range relative to the batch center (LO position)
        # These will be the actual frequencies we sweep in IF
        delta_freq_range = np.arange(
            batch_start - batch_center, 
            batch_end - batch_center, 
            freq_step
        )
        
        batches.append({
            'delta_freq_range': delta_freq_range,
            'lo_offset': lo_offset
        })
    
    return batches


def _acquisition(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy.
    
    Handles wideband spectroscopy by batching when the frequency range exceeds ±300 MHz from the LO
    """
    
    # Calculate batches 
    batches = _calculate_batches(params.freq_width, params.freq_step)
    
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
    
    # Storage for aggregated results across all batches
    all_frequencies = {qubit: [] for qubit in targets}
    all_signals = {qubit: [] for qubit in targets}
    all_phases = {qubit: [] for qubit in targets}
    all_error_signals = {qubit: [] for qubit in targets}
    all_error_phases = {qubit: [] for qubit in targets}
    amplitudes = {}
    
    # Execute each batch
    for batch_idx, batch in enumerate(batches):
        delta_frequency_range = batch['delta_freq_range']
        lo_offset = batch['lo_offset']
        
        # ----->>> Remover after tests
        if len(batches) > 1:
            print(f"Executing batch {batch_idx + 1}/{len(batches)}: "
                  f"LO offset = {lo_offset/1e6:.1f} MHz, "
                  f"sweep range = [{delta_frequency_range[0]/1e6:.1f}, {delta_frequency_range[-1]/1e6:.1f}] MHz")
        
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
                    values=f0 + lo_offset + delta_frequency_range,
                    channels=[qd_channel],
                )
            )
        
        # Prepare updates for this batch
        batch_updates = []
        for qubit in targets:
            update_dict = {
                platform.qubits[qubit].probe: {"frequency": readout_frequency(qubit, platform)}
            }
            
            # If we have an LO and we're batching, update it
            if lo_channels[qubit] is not None and lo_offset != 0:
                f0 = platform.config(drive_channels[qubit]).frequency
                new_lo_freq = f0 + lo_offset
                update_dict[lo_channels[qubit]] = {"frequency": new_lo_freq}
            
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
            all_frequencies[qubit].append(delta_frequency_range + f0 + lo_offset)
            all_signals[qubit].append(signal)
            all_phases[qubit].append(_phase)
            all_error_signals[qubit].append(error_signal)
            all_error_phases[qubit].append(error_phase)
    
    # Create data structure and aggregate results
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type, amplitudes=amplitudes
    )
    
    # Combine all batches for each qubit
    for qubit in targets:
        # Concatenate arrays from all batches
        freq = np.concatenate(all_frequencies[qubit])
        signal = np.concatenate(all_signals[qubit])
        _phase = np.concatenate(all_phases[qubit])
        
        # Handle errors
        if all_error_signals[qubit][0] is not None:
            error_signal = np.concatenate(all_error_signals[qubit])
            error_phase = np.concatenate(all_error_phases[qubit])
        else:
            error_signal = None
            error_phase = None
        
        # Sort by frequency (batches should already be in order, but just to be safe)
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        signal = signal[sort_idx]
        _phase = _phase[sort_idx]
        if error_signal is not None:
            error_signal = error_signal[sort_idx]
            error_phase = error_phase[sort_idx]
        
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
"""QubitSpectroscopy Routine object.

A qubit spectroscopy routine that sweeps the drive frequency around the qubit frequency at which the qubit's transition from |0> to |1> occurs.
Typically a long pulse is used to ensure a narrow frequency spectrum of the drive, and helping the qubit to get excited to a superposition state.
"""
