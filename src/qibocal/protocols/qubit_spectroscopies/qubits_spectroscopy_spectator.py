from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from qibocal.protocols.utils import HZ_TO_GHZ
from qibocal.result import magnitude, phase

__all__ = ["qubit_spectroscopy_spectator_scan"]

SpectatorFreqScanType = np.dtype(
    [
        ("t_freq", np.float64),
        ("t_signal", np.float64),
        ("t_phase", np.float64),
        ("s_freq", np.float64),
        ("s_signal", np.float64),
        ("s_phase", np.float64),
    ]
)


@dataclass
class SpectatorFreqScanParameters(QubitSpectroscopyParameters):
    """SpectatorFreqScan runcard inputs."""

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""


@dataclass
class SpectatorFreqScanResults(QubitSpectroscopyResults):
    """SpectatorFreqScan outputs."""

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""


@dataclass
class SpectatorFreqScanData(QubitSpectroscopyData):
    """SpectatorFreqScan acquisition outputs."""

    spectator_qubits: Optional[list[QubitId]] = None
    """List of spectator qubits to set in 1 state before qubit spectroscopy."""
    data: dict[QubitId, dict[QubitId, npt.NDArray[SpectatorFreqScanType]]] = field(
        default_factory=dict
    )
    "Raw data for SpectatorFreqScan."

    def register_qubit(
        self,
        target: QubitId,
        spectator: QubitId,
        target_freq: npt.NDArray,
        spectator_freq: npt.NDArray,
        target_signal: npt.NDArray,
        target_phase: npt.NDArray,
        spectator_signal: npt.NDArray,
        spectator_phase: npt.NDArray,
    ):
        """Create custom dtype array for acquired data."""
        size = target_freq.size * spectator_freq.size
        ar = np.empty(size, dtype=SpectatorFreqScanType)

        t_freqs, s_freqs = np.meshgrid(target_freq, spectator_freq)

        ar["t_freq"] = t_freqs.ravel()
        ar["t_signal"] = target_signal.ravel()
        ar["t_phase"] = target_phase.ravel()
        ar["s_freq"] = s_freqs.ravel()
        ar["s_signal"] = spectator_signal.ravel()
        ar["s_phase"] = spectator_phase.ravel()

        if target not in self.data:
            self.data[target] = {}
        self.data[target][spectator] = np.rec.array(ar)


def _acquisition(
    params: SpectatorFreqScanParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SpectatorFreqScanData:
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

    if any(x in targets for x in params.spectator_qubits):
        raise ValueError("One or multiple qubits are set as both spectator and target.")

    # Create data structure and aggregate results
    data = SpectatorFreqScanData(
        resonator_type=platform.resonator_type,
        targets=targets,
        amplitudes=amplitudes,
        spectator_qubits=params.spectator_qubits,
    )

    # executing the experiment for each spectator qubit separately
    for spect_q in params.spectator_qubits:
        spectator_seq, spectator_ro_pulses, spectator_drive_ch, spectator_los_ch, _ = (
            spectroscopy_sequence(
                params=params,
                platform=platform,
                targets=[spect_q],
            )
        )

        complete_seq = spectator_seq + spectro_seq

        # Execute each batch
        for start, end, lo_offset in batches:
            delta_frequency_range = np.arange(start, end, params.freq_step)

            # sweepers and updates for target qubits
            q_parsweep, q_batch_updates = create_spectr_sweeper_and_updates(
                platform=platform,
                targets=targets,
                drive_channels=drive_channels,
                delta_frequency_range=delta_frequency_range,
                los_channels=los_channels,
                lo_offset=lo_offset,
            )

            # sweepers and updates for spectators qubits
            spectator_parsweep, spectator_batch_updates = (
                create_spectr_sweeper_and_updates(
                    platform=platform,
                    targets=[spect_q],
                    drive_channels=spectator_drive_ch,
                    delta_frequency_range=delta_frequency_range,
                    los_channels=spectator_los_ch,
                    lo_offset=lo_offset,
                )
            )

            # Execute this batch
            results = platform.execute(
                [complete_seq],
                [spectator_parsweep, q_parsweep],
                updates=[q_batch_updates, spectator_batch_updates],
                **params.execution_parameters,
            )

            # Collect results from this batch
            spectator_result = results[spectator_ro_pulses[spect_q].id]
            spectator_signal = magnitude(spectator_result)
            spectator_phase = phase(spectator_result)

            for qubit in targets:
                result = results[ro_pulses[qubit].id]
                f0 = platform.config(drive_channels[qubit]).frequency

                signal = magnitude(result)
                _phase = phase(result)

                data.register_qubit(
                    target=qubit,
                    spectator=spect_q,
                    target_freq=delta_frequency_range + f0,
                    spectator_freq=spectator_parsweep[0].values,
                    target_signal=signal,
                    target_phase=_phase,
                    spectator_signal=spectator_signal,
                    spectator_phase=spectator_phase,
                )

    return data


def _fit(data: SpectatorFreqScanData) -> SpectatorFreqScanResults:
    """Post-processing function for SpectatorFreqScan."""
    return SpectatorFreqScanResults()


def _plot(data: SpectatorFreqScanData, target: QubitId, fit: SpectatorFreqScanResults):
    """Plotting function for SpectatorFreqScan."""

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
            go.Heatmap(
                x=data.data[target][spect]["t_freq"] * HZ_TO_GHZ,
                y=data.data[target][spect]["s_freq"] * HZ_TO_GHZ,
                z=data.data[target][spect]["t_signal"],
                name="Target Signal",
                legendgroup="Target Signal",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=data.data[target][spect]["t_freq"] * HZ_TO_GHZ,
                y=data.data[target][spect]["s_freq"] * HZ_TO_GHZ,
                z=data.data[target][spect]["t_phase"],
                name="Target Phase",
                legendgroup="Target Phase",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                x=data.data[target][spect]["t_freq"] * HZ_TO_GHZ,
                y=data.data[target][spect]["s_freq"] * HZ_TO_GHZ,
                z=data.data[target][spect]["s_signal"],
                name="Spectator Signal",
                legendgroup="Spectator Signal",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=data.data[target][spect]["t_freq"] * HZ_TO_GHZ,
                y=data.data[target][spect]["s_freq"] * HZ_TO_GHZ,
                z=data.data[target][spect]["s_phase"],
                name="Spectator Phase",
                legendgroup="Spectator Phase",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text=f"Spectroscopy 2D scan with {spect} as Spectator Qubit."
        )
        figures.append(fig)

    return figures, fitting_report


def _update(
    results: SpectatorFreqScanResults, platform: CalibrationPlatform, target: QubitId
):
    return


qubit_spectroscopy_spectator_scan = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
