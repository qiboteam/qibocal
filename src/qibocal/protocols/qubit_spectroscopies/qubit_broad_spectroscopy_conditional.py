from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Custom, Pulse, PulseSequence

from qibocal.auto.operation import QubitId, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.qubit_spectroscopies.utils import (
    QubitSpectroscopyData,
    QubitSpectroscopyParameters,
    QubitSpectroscopyResults,
    calculate_batches,
    create_spectr_sweeper_and_updates,
    spectroscopy_sequence,
)
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    lorentzian_fit,
)
from qibocal.result import magnitude, phase

__all__ = ["conditional_broad_spectator_spectroscopy"]


def envelope(duration: int, alpha: float, beta: float, interval: int):
    x = np.arange(duration)
    xp = x % interval / interval
    return np.cos(2 * np.pi * (alpha * xp + beta) * xp)


ConditionalBroadSpectatorSpectrumType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
        ("s_signal", np.float64),
        ("s_phase", np.float64),
    ]
)


@dataclass
class ConditionalBroadSpectatorSpectrumParameters(QubitSpectroscopyParameters):
    """ConditionalBroadSpectatorSpectrum runcard inputs."""

    spectator_pulse_amplitude: float = 0.0
    """Amplitude for spectator qubit pulse."""


@dataclass
class ConditionalBroadSpectatorSpectrumResults(QubitSpectroscopyResults):
    """ConditionalBroadSpectatorSpectrum outputs."""

    spectator_pulse_amplitude: float = field(default_factory=float)
    """Amplitude for spectator qubit pulse."""
    fitted_parameters: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Raw fitting output."""


@dataclass
class ConditionalBroadSpectatorSpectrumData(QubitSpectroscopyData):
    """ConditionalBroadSpectatorSpectrum acquisition outputs."""

    spectator_pulse_amplitude: float = field(default_factory=float)
    """Amplitude for spectator qubit pulse."""
    data: dict[QubitPairId, npt.NDArray[ConditionalBroadSpectatorSpectrumType]] = field(
        default_factory=dict
    )
    "Raw data for ConditionalBroadSpectatorSpectrum."

    def register_qubit(
        self,
        pair: QubitPairId,
        target_freq: npt.NDArray,
        target_signal: npt.NDArray,
        target_phase: npt.NDArray,
        spectator_signal: npt.NDArray,
        spectator_phase: npt.NDArray,
    ):
        """Create custom dtype array for acquired data."""
        ar = np.empty(target_freq.size, dtype=ConditionalBroadSpectatorSpectrumType)

        ar["freq"] = target_freq.ravel()
        ar["signal"] = target_signal.ravel()
        ar["phase"] = target_phase.ravel()
        ar["s_signal"] = spectator_signal.ravel()
        ar["s_phase"] = spectator_phase.ravel()

        self.data[pair] = np.rec.array(ar)


def _acquisition(
    params: ConditionalBroadSpectatorSpectrumParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ConditionalBroadSpectatorSpectrumData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by batching when the frequency range exceeds ±300 MHz from the LO
    """

    if any(isinstance(x, (str, int)) for x in targets):
        raise ValueError("At least one target is not a QubitPairId type.")

    target_qubits_list, spectator_qubits_list = map(list, zip(*targets))
    if any(s in target_qubits_list for s in spectator_qubits_list):
        raise ValueError("One or multiple qubits are set as both spectator and target.")

    # Calculate batches
    batches = calculate_batches(params.freq_width)

    spectro_seq, target_ro_seq, ro_pulses, drive_channels, los_channels, amplitudes = (
        spectroscopy_sequence(
            params=params,
            platform=platform,
            targets=target_qubits_list,
        )
    )

    # Create data structure and aggregate results
    data = ConditionalBroadSpectatorSpectrumData(
        resonator_type=platform.resonator_type,
        targets=targets,
        amplitudes=amplitudes,
        spectator_pulse_amplitude=params.spectator_pulse_amplitude,
    )

    # adding the spectators pulses
    spectator_ro_pulses = {}
    spectators_drive_sequence = PulseSequence()
    spectators_ro_sequence = PulseSequence()
    for spect_q in spectator_qubits_list:
        spect_natives = platform.natives.single_qubit[spect_q]

        spect_drive_channel, _ = spect_natives.RX()[0]
        spect_ro_channel, spect_ro_pulse = spect_natives.MZ()[0]
        spectator_ro_pulses[spect_q] = spect_ro_pulse

        custom_i = envelope(spectro_seq.duration, 8, 0, 500)
        spect_pulse = Pulse(
            amplitude=params.spectator_pulse_amplitude,
            duration=spectro_seq.duration,
            envelope=Custom(
                i_=custom_i,
                q_=np.zeros_like(custom_i),
            ),
        )

        # add custom pulse for the spectator qubit
        spectators_drive_sequence.append((spect_drive_channel, spect_pulse))
        # appending readout pulse for spectator qubit
        spectators_ro_sequence.append((spect_ro_channel, spect_ro_pulse))

    complete_sequence = spectators_drive_sequence + spectro_seq
    complete_sequence |= target_ro_seq + spectators_ro_sequence

    # Execute each batch
    for start, end, lo_offset in batches:
        delta_frequency_range = np.arange(start, end, params.freq_step)

        parsweep, batch_updates = create_spectr_sweeper_and_updates(
            platform=platform,
            targets=target_qubits_list,
            drive_channels=drive_channels,
            delta_frequency_range=delta_frequency_range,
            los_channels=los_channels,
            lo_offset=lo_offset,
        )

        # Execute this batch
        results = platform.execute(
            [complete_sequence],
            [list(parsweep.values())],
            updates=[batch_updates],
            **params.execution_parameters,
        )

        # Collect results from this batch
        for pair in targets:
            t, s = pair
            # Collect results for spectator qubits
            spectator_result = results[spectator_ro_pulses[s].id]
            spectator_signal = magnitude(spectator_result)
            spectator_phase = phase(spectator_result)

            result = results[ro_pulses[t].id]
            signal = magnitude(result)
            _phase = phase(result)

            data.register_qubit(
                pair=pair,
                target_freq=parsweep[t].values,
                target_signal=signal,
                target_phase=_phase,
                spectator_signal=spectator_signal,
                spectator_phase=spectator_phase,
            )

    return data


def _fit(
    data: ConditionalBroadSpectatorSpectrumData,
) -> ConditionalBroadSpectatorSpectrumResults:
    """Post-processing function for ConditionalBroadSpectatorSpectrum."""
    frequency = {}
    fitted_parameters = {}
    for pair in data.data:
        frequency[pair[0]] = {}
        fitted_parameters[pair] = {}
        fit_result = lorentzian_fit(
            data[pair], resonator_type=data.resonator_type, fit="qubit"
        )
        if fit_result is not None:
            fit_freq, fit_params, _ = fit_result

            frequency[pair[0]] = fit_freq
            fitted_parameters[pair] = fit_params

    return ConditionalBroadSpectatorSpectrumResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
        spectator_pulse_amplitude=data.spectator_pulse_amplitude,
    )


def _plot(
    data: ConditionalBroadSpectatorSpectrumData,
    target: QubitPairId,
    fit: ConditionalBroadSpectatorSpectrumResults,
):
    """Plotting function for ConditionalBroadSpectatorSpectrum."""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.4,
        subplot_titles=(
            "Target Signal [a.u.]",
            "Target Phase [rad]",
            "Spectator Signal [a.u.]",
            "Spectator Phase [rad]",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.data[target].freq * HZ_TO_GHZ,
            y=data.data[target].signal,
            name="Target Signal",
            legendgroup="Target Signal",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.data[target].freq * HZ_TO_GHZ,
            y=data.data[target].phase,
            name="Target Phase",
            legendgroup="Target Phase",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data.data[target].freq * HZ_TO_GHZ,
            y=data.data[target].s_signal,
            name="Spectator Signal",
            legendgroup="Spectator Signal",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.data[target].freq * HZ_TO_GHZ,
            y=data.data[target].s_phase,
            name="Spectator Phase",
            legendgroup="Spectator Phase",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(title_text=f"Spectroscopy with {target[1]} as Spectator Qubit.")
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ConditionalBroadSpectatorSpectrumResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    return


conditional_broad_spectator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
