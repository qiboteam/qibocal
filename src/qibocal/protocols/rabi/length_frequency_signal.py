"""Rabi experiment that sweeps length and frequency."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ..utils import HZ_TO_GHZ, fallback_period, guess_period
from .length_signal import RabiLengthSignalResults
from .utils import fit_length_function, sequence_length


@dataclass
class RabiLengthFrequencySignalParameters(Parameters):
    """RabiLengthFrequency runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""


@dataclass
class RabiLengthFrequencySignalResults(RabiLengthSignalResults):
    """RabiLengthFrequency outputs."""

    frequency: dict[QubitId, Union[float, list[float]]]
    """Drive frequency for each qubit."""


RabiLenFreqSignalType = np.dtype(
    [
        ("len", np.float64),
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqSignalData(Data):
    """RabiLengthFreqSignal data acquisition."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenFreqSignalType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, lens, signal, phase):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        frequency, length = np.meshgrid(freq, lens)
        data = np.empty(size, dtype=RabiLenFreqSignalType)
        data["freq"] = frequency.ravel()
        data["len"] = length.ravel()
        data["signal"] = signal.ravel()
        data["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(data)

    def durations(self, qubit):
        """Unique qubit lengths."""
        return np.unique(self[qubit].len)

    def frequencies(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)


def _acquisition(
    params: RabiLengthFrequencySignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RabiLengthFreqSignalData:
    """Data acquisition for Rabi experiment sweeping length."""

    sequence, qd_pulses, ro_pulses, amplitudes = sequence_length(
        targets, params, platform
    )

    # qubit drive pulse length
    length_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    sweeper_len = Sweeper(
        Parameter.duration,
        length_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.ABSOLUTE,
    )

    # qubit drive pulse amplitude
    frequency_range = np.arange(
        params.min_freq,
        params.max_freq,
        params.step_freq,
    )
    sweeper_freq = Sweeper(
        Parameter.frequency,
        frequency_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )

    data = RabiLengthFreqSignalData(amplitudes=amplitudes)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_len,
        sweeper_freq,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit,
            freq=qd_pulses[qubit].frequency + frequency_range,
            lens=length_range,
            signal=result.magnitude,
            phase=result.phase,
        )
    return data


def _fit(data: RabiLengthFreqSignalData) -> RabiLengthFrequencySignalResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_durations = {}
    fitted_parameters = {}

    for qubit in data.data:
        durations = data.durations(qubit)
        freqs = data.frequencies(qubit)
        signal = data[qubit].signal
        signal_matrix = signal.reshape(len(durations), len(freqs)).T

        # guess optimal frequency maximizing oscillatio amplitude
        index = np.argmax([max(x) - min(x) for x in signal_matrix])
        frequency = freqs[index]

        y = signal_matrix[index]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(durations)
        x_max = np.max(durations)
        x = (durations - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        period = fallback_period(guess_period(x, y))
        pguess = [0, np.sign(y[0]) * 0.5, period, 0, 0]

        try:
            popt, _, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            fitted_frequencies[qubit] = frequency
            fitted_durations[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthFrequencySignalResults(
        length=fitted_durations,
        amplitude=data.amplitudes,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
    )


def _plot(
    data: RabiLengthFreqSignalData,
    target: QubitId,
    fit: RabiLengthFrequencySignalResults = None,
):
    """Plotting function for RabiLengthFrequency."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Signal [a.u.]",
            "Phase [rad]",
        ),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    durations = qubit_data.len

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Durations [ns]", row=1, col=1)
    fig.update_xaxes(title_text="Durations [ns]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[min(durations), max(durations)],
                y=[fit.frequency[target] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[min(durations), max(durations)],
                y=[fit.frequency[target] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
            row=1,
            col=2,
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", "Pi-pulse duration"],
                [
                    fit.frequency[target],
                    f"{fit.length[target]:.2f} ns",
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )

    return figures, fitting_report


def _update(
    results: RabiLengthFrequencySignalResults, platform: Platform, target: QubitId
):
    update.drive_amplitude(results.amplitude[target], platform, target)
    update.drive_duration(results.length[target], platform, target)
    update.drive_frequency(results.frequency[target], platform, target)


rabi_length_frequency_signal = Routine(_acquisition, _fit, _plot, _update)
"""Rabi length with frequency tuning."""
