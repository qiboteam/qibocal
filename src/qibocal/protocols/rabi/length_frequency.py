from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ..two_qubit_interaction.utils import fit_flux_amplitude
from ..utils import HZ_TO_GHZ
from .length_signal import RabiLengthVoltResults
from .utils import period_correction_factor, rabi_length_function


@dataclass
class RabiLengthFrequencyParameters(Parameters):
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
class RabiLengthFrequencyResults(RabiLengthVoltResults):
    """RabiLengthFrequency outputs."""

    frequency: dict[QubitId, tuple[float, Optional[int]]]
    """Drive frequency for each qubit."""


RabiLenFreqVoltType = np.dtype(
    [
        ("len", np.float64),
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqVoltData(Data):
    """RabiLengthFreqVolt data acquisition."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenFreqVoltType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, lens, signal, phase):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        frequency, length = np.meshgrid(freq, lens)
        ar = np.empty(size, dtype=RabiLenFreqVoltType)
        ar["freq"] = frequency.ravel()
        ar["len"] = length.ravel()
        ar["signal"] = signal.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)

    def durations(self, qubit):
        """Unique qubit lengths."""
        return np.unique(self[qubit].len)

    def frequencies(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)


def _acquisition(
    params: RabiLengthFrequencyParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthFreqVoltData:
    """Data acquisition for Rabi experiment sweeping length."""

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude

        amplitudes[qubit] = qd_pulses[qubit].amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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

    data = RabiLengthFreqVoltData(amplitudes=amplitudes)

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


def _fit(data: RabiLengthFreqVoltData) -> RabiLengthFrequencyResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_durations = {}
    fitted_parameters = {}

    for qubit in data.data:
        durations = data.durations(qubit)
        freqs = data.frequencies(qubit)
        signal = data[qubit].signal
        signal_matrix = signal.reshape(len(durations), len(freqs)).T

        # guess amplitude computing FFT
        frequency, index, _ = fit_flux_amplitude(signal_matrix, freqs, durations)

        y = signal_matrix[index, :].ravel()
        pguess = [0, np.sign(y[0]) * 0.5, 1 / frequency, 0, 0]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(durations)
        x_max = np.max(durations)
        x = (durations - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        try:
            popt, _ = curve_fit(
                rabi_length_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, -1, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )
            translated_popt = [  # change it according to the fit function
                (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = (
                translated_popt[2]
                / 2
                * period_correction_factor(phase=translated_popt[3])
            )
            fitted_frequencies[qubit] = frequency
            fitted_durations[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = translated_popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthFrequencyResults(
        length=fitted_durations,
        amplitude=data.amplitudes,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
    )


def _plot(
    data: RabiLengthFreqVoltData,
    target: QubitId,
    fit: RabiLengthFrequencyResults = None,
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
            "Normalised Signal [a.u.]",
            "phase [rad]",
        ),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    durations = qubit_data.len

    # n_amps = len(np.unique(qubit_data.amp))
    # n_freq = len(np.unique(qubit_data.freq))
    # for i in range(n_amps):
    # qubit_data.signal[i * n_freq : (i + 1) * n_freq] = norm(
    #    qubit_data.signal[i * n_freq : (i + 1) * n_freq]
    # )

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

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Durations [ns]", row=1, col=1)
    fig.update_xaxes(title_text="Durations [ns]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target,
                ["Transition frequency", "Pi-pulse duration"],
                [
                    fit.frequency[target],
                    fit.length[target],
                ],
            )
        )
    return figures, fitting_report


rabi_length_frequency = Routine(_acquisition, _fit, _plot)
"""Rabi length with frequency tuning."""
