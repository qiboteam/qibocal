"""Rabi experiment that sweeps length and frequency."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ...result import unpack
from ..utils import HZ_TO_GHZ, readout_frequency
from .utils import fit_length_function, rabi_initial_guess, sequence_length

__all__ = ["conditional_rabi_chevron_len_signal"]


@dataclass
class ConditionalRabiChevronLenSignalParameters(Parameters):
    """ConditionalRabiChevron runcard inputs."""

    pulse_duration_start: int
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: int
    """Final pi pulse duration [ns]."""
    pulse_duration_step: int
    """Step pi pulse duration [ns]."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""

    @property
    def frequency_range(self):
        return np.arange(
            self.min_freq,
            self.max_freq,
            self.step_freq,
        )

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start,
            self.pulse_duration_end,
            self.pulse_duration_step,
        )


@dataclass
class ConditionalRabiChevronLenSignalResults(Results):
    """ConditionalRabiChevron outputs."""

    length: dict[QubitPairId, Union[int, list[int]]]
    """Pi pulse duration for each qubit."""
    fitted_parameters: dict[QubitPairId, dict[str, float]]
    """Raw fitting output."""


CondRabiLenSigChevronType = np.dtype(
    [
        ("len", int),
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqSignalData(Data):
    """RabiLengthFreqSignal data acquisition."""

    data: dict[QubitPairId, npt.NDArray[CondRabiLenSigChevronType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, pair, freq, lens, i, q):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        length, frequency = np.meshgrid(lens, freq)
        data = np.empty(size, dtype=CondRabiLenSigChevronType)
        data["freq"] = frequency.ravel()
        data["len"] = length.ravel()
        data["i"] = i.ravel()
        data["q"] = q.ravel()
        self.data[pair] = np.rec.array(data)

    def durations(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Unique qubit lengths."""
        return np.unique(self[qubit_pair].len)

    def frequencies(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Unique qubit frequency."""
        return np.unique(self[qubit_pair].freq)

    def compute_magnitude(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Compute magnitude of the measured signal"""
        return np.sqrt(self[qubit_pair].i ** 2 + self[qubit_pair].q ** 2)

    def compute_phase(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Compute phase of the measured signal"""
        return np.unwrap(np.arctan2(self[qubit_pair].i, self[qubit_pair].q))


def _acquisition(
    params: ConditionalRabiChevronLenSignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> RabiLengthFreqSignalData:
    """Data acquisition for Rabi experiment sweeping length."""

    if any(isinstance(x, (str, int)) for x in targets):
        raise ValueError("At least one target is not a QubitPairId type.")

    target_qubits_list, spectator_qubits_list = map(list, zip(*targets))
    if any(s in target_qubits_list for s in spectator_qubits_list):
        raise ValueError("One or multiple qubits are set as both spectator and target.")

    t_sequence, t_qd_pulses, t_delays, t_ro_pulses, _ = sequence_length(
        target_qubits_list, params, platform, False
    )

    complete_sequence = PulseSequence()
    for s in spectator_qubits_list:
        spectator_natives = platform.natives.single_qubit[s]
        spectator_ch, spectator_pulse = spectator_natives.RX()[0]

        complete_sequence.append((spectator_ch, spectator_pulse))

    complete_seq_duration = complete_sequence.duration
    complete_sequence += PulseSequence(
        (ch, Delay(duration=complete_seq_duration)) for ch in t_sequence.channels
    )
    complete_sequence += t_sequence

    len_sweeper = Sweeper(
        parameter=Parameter.duration,
        values=params.duration_range,
        pulses=[t_qd_pulses[q] for q in target_qubits_list]
        + [t_delays[q] for q in target_qubits_list],
    )

    freq_sweepers = {}
    for t in target_qubits_list:
        target_drive_ch = platform.qubits[t].drive
        freq_sweepers[t] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(target_drive_ch).frequency + params.frequency_range,
            channels=[target_drive_ch],
        )

    data = RabiLengthFreqSignalData()

    results = platform.execute(
        [complete_sequence],
        [list(freq_sweepers.values()), [len_sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in target_qubits_list
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for pair in targets:
        result = results[t_ro_pulses[pair[0]].id]
        i, q = unpack(result)

        data.register_qubit(
            pair=pair,
            freq=freq_sweepers[pair[0]].values,
            lens=len_sweeper.values,
            i=i,
            q=q,
        )
    return data


def _fit(data: RabiLengthFreqSignalData) -> ConditionalRabiChevronLenSignalResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_durations = {}
    fitted_parameters = {}

    for pair in data.data:
        durations = data.durations(pair)
        freqs = data.frequencies(pair)
        mag = data.compute_magnitude(pair)
        mag_matrix = mag.reshape(len(durations), len(freqs)).T

        # guess optimal frequency maximizing oscillatio amplitude
        index = np.argmax([max(x) - min(x) for x in mag_matrix])
        frequency = freqs[index]

        y = mag_matrix[index]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(durations)
        x_max = np.max(durations)
        x = (durations - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        pguess = rabi_initial_guess(x, y, "length", signal=True)

        try:
            popt, _, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            fitted_frequencies[pair] = frequency
            fitted_durations[pair] = int(pi_pulse_parameter)
            fitted_parameters[pair] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for pair {pair} due to {e}.")

    return ConditionalRabiChevronLenSignalResults(
        length=fitted_durations,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: RabiLengthFreqSignalData,
    target: QubitPairId,
    fit: ConditionalRabiChevronLenSignalResults = None,
):
    """Plotting function for ConditionalRabiChevron."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            f"Signal [a.u.] - Target {target[0]}, Spectator {target[1]}",
            f"Phase [rad] - Target {target[0]}, Spectator {target[1]}",
        ),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    durations = qubit_data.len
    mag = data.compute_magnitude(target)
    phase = data.compute_phase(target)

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=mag,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Durations [ns]", row=1, col=1)
    fig.update_xaxes(title_text="Durations [ns]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )

    return figures, fitting_report


def _update(
    results: ConditionalRabiChevronLenSignalResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    return


conditional_rabi_chevron_len_signal = Routine(_acquisition, _fit, _plot, _update)
"""Rabi length with frequency tuning."""
