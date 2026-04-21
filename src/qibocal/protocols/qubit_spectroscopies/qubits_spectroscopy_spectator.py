from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.auto.operation import QubitPairId, Routine
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
from qibocal.result import unpack

__all__ = ["qubit_spectroscopy_spectator_scan"]

SpectatorFreqScanType = np.dtype(
    [
        ("t_freq", np.float64),
        ("t_i", np.float64),
        ("t_q", np.float64),
        ("s_freq", np.float64),
        ("s_i", np.float64),
        ("s_q", np.float64),
    ]
)


@dataclass
class SpectatorFreqScanParameters(QubitSpectroscopyParameters):
    """SpectatorFreqScan runcard inputs."""


@dataclass
class SpectatorFreqScanResults(QubitSpectroscopyResults):
    """SpectatorFreqScan outputs."""


@dataclass
class SpectatorFreqScanData(QubitSpectroscopyData):
    """SpectatorFreqScan acquisition outputs."""

    data: dict[QubitPairId, npt.NDArray[SpectatorFreqScanType]] = field(
        default_factory=dict
    )
    "Raw data for SpectatorFreqScan."

    def register_qubit(
        self,
        pair: QubitPairId,
        target_freq: npt.NDArray,
        spectator_freq: npt.NDArray,
        target_i: npt.NDArray,
        target_q: npt.NDArray,
        spectator_i: npt.NDArray,
        spectator_q: npt.NDArray,
    ):
        """Create custom dtype array for acquired data."""
        size = target_freq.size * spectator_freq.size
        ar = np.empty(size, dtype=SpectatorFreqScanType)

        t_freqs, s_freqs = np.meshgrid(target_freq, spectator_freq)

        ar["t_freq"] = t_freqs.ravel()
        ar["t_i"] = target_i.ravel()
        ar["t_q"] = target_q.ravel()
        ar["s_freq"] = s_freqs.ravel()
        ar["s_i"] = spectator_i.ravel()
        ar["s_q"] = spectator_q.ravel()

        self.data[pair] = np.rec.array(ar)

    def compute_magnitude(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Compute magnitude of the measured signal"""
        return np.sqrt(self[qubit_pair].i ** 2 + self[qubit_pair].q ** 2)

    def compute_phase(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Compute phase of the measured signal"""
        return np.unwrap(np.arctan2(self[qubit_pair].i, self[qubit_pair].q))


def _acquisition(
    params: SpectatorFreqScanParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> SpectatorFreqScanData:
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
    data = SpectatorFreqScanData(
        resonator_type=platform.resonator_type,
        targets=targets,
        amplitudes=amplitudes,
    )

    # executing the experiment for each spectator qubit separately
    spectator_seq, spectator_ro_pulses, spectator_drive_ch, spectator_los_ch, _ = (
        spectroscopy_sequence(
            params=params,
            platform=platform,
            targets=spectator_qubits_list,
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
        spectator_parsweep, spectator_batch_updates = create_spectr_sweeper_and_updates(
            platform=platform,
            targets=spectator_qubits_list,
            drive_channels=spectator_drive_ch,
            delta_frequency_range=delta_frequency_range,
            los_channels=spectator_los_ch,
            lo_offset=lo_offset,
        )

        # Execute this batch
        results = platform.execute(
            [complete_seq],
            [list(spectator_parsweep.values()), list(q_parsweep.values())],
            updates=[q_batch_updates, spectator_batch_updates],
            **params.execution_parameters,
        )

        # Collect results from this batch
        for pair in targets:
            targ, spect = pair

            target_result = results[ro_pulses[targ].id]
            target_i, target_q = unpack(target_result)

            spectator_result = results[spectator_ro_pulses[spect].id]
            spectator_i, spectator_q = unpack(spectator_result)

            data.register_qubit(
                pair=pair,
                target_freq=q_parsweep[targ].values,
                spectator_freq=spectator_parsweep[spect].values,
                target_i=target_i,
                target_q=target_q,
                spectator_i=spectator_i,
                spectator_q=spectator_q,
            )

    return data


def _fit(data: SpectatorFreqScanData) -> SpectatorFreqScanResults:
    """Post-processing function for SpectatorFreqScan."""
    return SpectatorFreqScanResults()


def _plot(
    data: SpectatorFreqScanData, target: QubitPairId, fit: SpectatorFreqScanResults
):
    """Plotting function for SpectatorFreqScan."""

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
        go.Heatmap(
            x=data[target].t_freq * HZ_TO_GHZ,
            y=data[target].s_freq * HZ_TO_GHZ,
            z=data.compute_magnitude(target),
            name="Target Signal",
            legendgroup="Target Signal",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=data[target].t_freq * HZ_TO_GHZ,
            y=data[target].s_freq * HZ_TO_GHZ,
            z=data.compute_phase(target),
            name="Target Phase",
            legendgroup="Target Phase",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            x=data[target].t_freq * HZ_TO_GHZ,
            y=data[target].s_freq * HZ_TO_GHZ,
            z=data.compute_magnitude(target),
            name="Spectator Signal",
            legendgroup="Spectator Signal",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=data[target].t_freq * HZ_TO_GHZ,
            y=data[target].s_freq * HZ_TO_GHZ,
            z=data.compute_phase(target),
            name="Spectator Phase",
            legendgroup="Spectator Phase",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text=f"Spectroscopy 2D scan with {target[1]} as Spectator Qubit."
    )
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: SpectatorFreqScanResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    return


qubit_spectroscopy_spectator_scan = Routine(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
