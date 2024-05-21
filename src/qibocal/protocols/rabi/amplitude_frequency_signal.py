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

from ..utils import HZ_TO_GHZ
from .amplitude_signal import RabiAmplitudeVoltResults
from .utils import rabi_amplitude_function


@dataclass
class RabiAmplitudeFrequencyVoltParameters(Parameters):
    """RabiAmplitudeFrequency runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    pulse_length: Optional[float] = None
    """RX pulse duration [ns]."""


@dataclass
class RabiAmplitudeFrequencyVoltResults(RabiAmplitudeVoltResults):
    """RabiAmplitudeFrequency outputs."""

    frequency: dict[QubitId, tuple[float, Optional[int]]]
    """Drive frequency for each qubit."""


RabiAmpFreqVoltType = np.dtype(
    [
        ("amp", np.float64),
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeFreqVoltData(Data):
    """RabiAmplitudeFreqVolt data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpFreqVoltType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, signal, phase):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        ar = np.empty(size, dtype=RabiAmpFreqVoltType)
        ar["freq"] = frequency.ravel()
        ar["amp"] = amplitude.ravel()
        ar["signal"] = signal.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)

    def amplitudes(self, qubit):
        """Unique qubit amplitudes."""
        return np.unique(self[qubit].amp)

    def frequencies(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)


def _acquisition(
    params: RabiAmplitudeFrequencyVoltParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RabiAmplitudeFreqVoltData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # qubit drive pulse amplitude
    amplitude_range = np.arange(
        params.min_amp_factor,
        params.max_amp_factor,
        params.step_amp_factor,
    )
    sweeper_amp = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.FACTOR,
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

    data = RabiAmplitudeFreqVoltData(durations=durations)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_amp,
        sweeper_freq,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit,
            freq=qd_pulses[qubit].frequency + frequency_range,
            amp=qd_pulses[qubit].amplitude * amplitude_range,
            signal=result.magnitude,
            phase=result.phase,
        )
    return data


def _fit(data: RabiAmplitudeFreqVoltData) -> RabiAmplitudeFrequencyVoltResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_amplitudes = {}
    fitted_parameters = {}

    for qubit in data.data:
        amps = data.amplitudes(qubit)
        freqs = data.frequencies(qubit)
        signal = data[qubit].signal
        signal_matrix = signal.reshape(len(amps), len(freqs)).T

        # guess optimal frequency maximizing oscillatio amplitude
        index = np.argmax([max(x) - min(x) for x in signal_matrix])
        frequency = freqs[index]

        pguess = [0.5, 1, 1 / frequency, 0]
        y = signal_matrix[index, :].ravel()

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(amps)
        x_max = np.max(amps)
        x = (amps - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        try:
            popt, _ = curve_fit(
                rabi_amplitude_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
            )
            translated_popt = [  # Change it according to fit function changes
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / (x_max - x_min) / popt[2],
            ]
            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = translated_popt[1]
            fitted_parameters[qubit] = translated_popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeFrequencyVoltResults(
        amplitude=fitted_amplitudes,
        length=data.durations,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
    )


def _plot(
    data: RabiAmplitudeFreqVoltData,
    target: QubitId,
    fit: RabiAmplitudeFrequencyVoltResults = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
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
    amplitudes = qubit_data.amp

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[min(amplitudes), max(amplitudes)],
            y=[fit.frequency[target] / 1e9] * 2,
            mode="lines",
            line=dict(color="white", width=4, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[min(amplitudes), max(amplitudes)],
            y=[fit.frequency[target] / 1e9] * 2,
            mode="lines",
            line=dict(color="white", width=4, dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=False,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", "Pi-pulse amplitude"],
                [
                    fit.frequency[target],
                    fit.amplitude[target],
                ],
            )
        )
    return figures, fitting_report


rabi_amplitude_frequency_signal = Routine(_acquisition, _fit, _plot)
"""Rabi amplitude with frequency tuning."""
