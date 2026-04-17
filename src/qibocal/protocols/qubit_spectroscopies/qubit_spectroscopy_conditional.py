from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Delay, PulseSequence

from qibocal.auto.operation import QubitId, Routine
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

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""


@dataclass
class ConditionalSpectrumResults(QubitSpectroscopyResults):
    """ConditionalSpectrum outputs."""

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""
    fitted_parameters: dict[QubitId, dict[QubitId, list[float]]] = field(
        default_factory=dict
    )
    """Raw fitting output."""


@dataclass
class ConditionalSpectrumData(QubitSpectroscopyData):
    """ConditionalSpectrum acquisition outputs."""

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""
    data: dict[QubitId, dict[QubitId, npt.NDArray[ConditionalSpectrumType]]] = field(
        default_factory=dict
    )
    "Raw data for ConditionalSpectrum."

    def register_qubit(
        self,
        target: QubitId,
        spectator: QubitId,
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

        if target not in self.data:
            self.data[target] = {}
        self.data[target][spectator] = np.rec.array(ar)


def _acquisition(
    params: ConditionalSpectrumParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ConditionalSpectrumData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by batching when the frequency range exceeds ±300 MHz from the LO
    """

    if params.spectator_qubits is None:
        raise ValueError("No spectator qubit was inserted.")

    # Calculate batches
    batches = calculate_batches(params.freq_width)

    spectro_seq, ro_pulses, drive_channels, los_channels, amplitudes = (
        spectroscopy_sequence(
            params=params,
            platform=platform,
            targets=targets,
        )
    )

    # Create data structure and aggregate results
    data = ConditionalSpectrumData(
        resonator_type=platform.resonator_type,
        targets=targets,
        amplitudes=amplitudes,
        spectator_qubits=params.spectator_qubits,
    )

    # adding the spectators pulses
    for spect_q in params.spectator_qubits:
        spect_natives = platform.natives.single_qubit[spect_q]

        spect_drive_channel, spect_pulse = spect_natives.RX()[0]
        spect_ro_channel, spect_ro_pulse = spect_natives.MZ()[0]

        # add pi-pulse for the spectator qubit
        spectator_seq = PulseSequence(((spect_drive_channel, spect_pulse),))
        # adding delay for target channels
        spectator_seq += PulseSequence(
            (ch, Delay(duration=spectator_seq.duration)) for ch in spectro_seq.channels
        )
        complete_sequence = spectator_seq + spectro_seq
        # appending readout pulse for spectator qubit
        spect_ro_sequence = PulseSequence(
            (
                (spect_ro_channel, Delay(duration=complete_sequence.duration)),
                (spect_ro_channel, spect_ro_pulse),
            )
        )
        complete_sequence += spect_ro_sequence

        # Execute each batch
        for start, end, lo_offset in batches:
            delta_frequency_range = np.arange(start, end, params.freq_step)

            parsweep, batch_updates = create_spectr_sweeper_and_updates(
                platform=platform,
                targets=targets,
                drive_channels=drive_channels,
                delta_frequency_range=delta_frequency_range,
                los_channels=los_channels,
                lo_offset=lo_offset,
            )

            # Execute this batch
            results = platform.execute(
                [complete_sequence],
                [parsweep],
                updates=[batch_updates],
                **params.execution_parameters,
            )

            # Collect results for spectator qubits
            spectator_result = results[spect_ro_pulse.id]
            spectator_signal = magnitude(spectator_result)
            spectator_phase = phase(spectator_result)

            # Collect results from this batch
            for qubit in targets:
                result = results[ro_pulses[qubit].id]
                f0 = platform.config(drive_channels[qubit]).frequency

                signal = magnitude(result)
                _phase = phase(result)

                data.register_qubit(
                    target=qubit,
                    spectator=spect_q,
                    target_freq=delta_frequency_range + f0,
                    target_signal=signal,
                    target_phase=_phase,
                    spectator_signal=spectator_signal,
                    spectator_phase=spectator_phase,
                )

    return data


def _fit(data: ConditionalSpectrumData) -> ConditionalSpectrumResults:
    """Post-processing function for ConditionalSpectrum."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        frequency[qubit] = {}
        fitted_parameters[qubit] = {}
        for spect in data.spectator_qubits:
            fit_result = lorentzian_fit(
                data[qubit][spect], resonator_type=data.resonator_type, fit="qubit"
            )
            if fit_result is not None:
                fit_freq, fit_params, _ = fit_result

                frequency[qubit][spect] = fit_freq
                fitted_parameters[qubit][spect] = fit_params

    return ConditionalSpectrumResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
        spectator_qubits=data.spectator_qubits,
    )


def _plot(
    data: ConditionalSpectrumData, target: QubitId, fit: ConditionalSpectrumResults
):
    """Plotting function for ConditionalSpectrum."""

    figures = []
    fitting_report = ""

    for spect in data.spectator_qubits:
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
                x=data.data[target][spect].freq * HZ_TO_GHZ,
                y=data.data[target][spect].signal,
                name="Target Signal",
                legendgroup="Target Signal",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.data[target][spect].freq * HZ_TO_GHZ,
                y=data.data[target][spect].phase,
                name="Target Phase",
                legendgroup="Target Phase",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=data.data[target][spect].freq * HZ_TO_GHZ,
                y=data.data[target][spect].s_signal,
                name="Spectator Signal",
                legendgroup="Spectator Signal",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.data[target][spect].freq * HZ_TO_GHZ,
                y=data.data[target][spect].s_phase,
                name="Spectator Phase",
                legendgroup="Spectator Phase",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title_text=f"Spectroscopy with {spect} as Spectator Qubit.")
        figures.append(fig)

    return figures, fitting_report


def _update(
    results: ConditionalSpectrumResults, platform: CalibrationPlatform, target: QubitId
):
    return


qubit_conditional_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
