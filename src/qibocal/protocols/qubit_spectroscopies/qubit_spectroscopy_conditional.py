from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Delay, PulseSequence

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

__all__ = ["qubit_conditional_spectroscopy"]


ConditionalSpectrumType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
        ("s_signal", np.float64),
        ("s_phase", np.float64),
    ]
)


@dataclass
class ConditionalSpectrumParameters(QubitSpectroscopyParameters):
    """ConditionalSpectrum runcard inputs."""


@dataclass
class ConditionalSpectrumResults(QubitSpectroscopyResults):
    """ConditionalSpectrum outputs."""

    fitted_parameters: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Raw fitting output."""


@dataclass
class ConditionalSpectrumData(QubitSpectroscopyData):
    """ConditionalSpectrum acquisition outputs."""

    data: dict[QubitPairId, npt.NDArray[ConditionalSpectrumType]] = field(
        default_factory=dict
    )
    "Raw data for ConditionalSpectrum."

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
        ar = np.empty(target_freq.size, dtype=ConditionalSpectrumType)

        ar["freq"] = target_freq.ravel()
        ar["signal"] = target_signal.ravel()
        ar["phase"] = target_phase.ravel()
        ar["s_signal"] = spectator_signal.ravel()
        ar["s_phase"] = spectator_phase.ravel()

        self.data[pair] = np.rec.array(ar)


def _acquisition(
    params: ConditionalSpectrumParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ConditionalSpectrumData:
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

    spectro_seq, ro_pulses, drive_channels, los_channels, amplitudes = (
        spectroscopy_sequence(
            params=params,
            platform=platform,
            targets=target_qubits_list,
        )
    )

    # Create data structure and aggregate results
    data = ConditionalSpectrumData(
        resonator_type=platform.resonator_type,
        targets=targets,
        amplitudes=amplitudes,
    )

    # adding the spectators pulses
    spectator_ro_pulses = {}
    spectators_drive_sequence = PulseSequence()
    spectators_ro_sequence = PulseSequence()
    for spect_q in spectator_qubits_list:
        spect_natives = platform.natives.single_qubit[spect_q]

        spect_drive_channel, spect_pulse = spect_natives.RX()[0]
        spect_ro_channel, spect_ro_pulse = spect_natives.MZ()[0]

        spectator_ro_pulses[spect_q] = spect_ro_pulse

        # add pi-pulse for the spectator qubit
        spectators_drive_sequence.append((spect_drive_channel, spect_pulse))
        # appending readout pulse for spectator qubit
        spectators_ro_sequence.append(
            (
                spect_ro_channel,
                Delay(
                    duration=(spectro_seq.duration + spectators_drive_sequence.duration)
                ),
            )
        )
        spectators_ro_sequence.append((spect_ro_channel, spect_ro_pulse))

    # adding delay for target channels
    spect_drive_seq_duration = spectators_drive_sequence.duration
    spectators_drive_sequence += PulseSequence(
        (ch, Delay(duration=spect_drive_seq_duration)) for ch in spectro_seq.channels
    )

    complete_sequence = spectators_drive_sequence + spectro_seq + spectators_ro_sequence

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
        for t, s in targets:
            # Collect results for spectator qubits
            spectator_result = results[spectator_ro_pulses[s].id]
            spectator_signal = magnitude(spectator_result)
            spectator_phase = phase(spectator_result)

            result = results[ro_pulses[t].id]
            signal = magnitude(result)
            _phase = phase(result)

            data.register_qubit(
                target=t,
                spectator=s,
                target_freq=parsweep[t],
                target_signal=signal,
                target_phase=_phase,
                spectator_signal=spectator_signal,
                spectator_phase=spectator_phase,
            )

    return data


def _fit(data: ConditionalSpectrumData) -> ConditionalSpectrumResults:
    """Post-processing function for ConditionalSpectrum."""
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

    return ConditionalSpectrumResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
    )


def _plot(
    data: ConditionalSpectrumData, target: QubitPairId, fit: ConditionalSpectrumResults
):
    """Plotting function for ConditionalSpectrum."""

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
    results: ConditionalSpectrumResults, platform: CalibrationPlatform, target: QubitId
):
    return


qubit_conditional_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
