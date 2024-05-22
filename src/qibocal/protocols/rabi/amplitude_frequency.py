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
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ..utils import HZ_TO_GHZ, chi2_reduced
from .amplitude_frequency_signal import (
    RabiAmplitudeFrequencyVoltParameters,
    RabiAmplitudeFrequencyVoltResults,
    _update,
)
from .utils import period_correction_factor, rabi_amplitude_function


@dataclass
class RabiAmplitudeFrequencyParameters(RabiAmplitudeFrequencyVoltParameters):
    """RabiAmplitudeFrequency runcard inputs."""


@dataclass
class RabiAmplitudeFrequencyResults(RabiAmplitudeFrequencyVoltResults):
    """RabiAmplitudeFrequency outputs."""

    chi2: dict[QubitId, tuple[float, Optional[float]]] = field(default_factory=dict)


RabiAmpFreqType = np.dtype(
    [
        ("amp", np.float64),
        ("freq", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeFreqData(Data):
    """RabiAmplitudeFreq data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpFreqType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, prob, error):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        ar = np.empty(size, dtype=RabiAmpFreqType)
        ar["freq"] = frequency.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = np.array(prob).ravel()
        ar["error"] = np.array(error).ravel()
        self.data[qubit] = np.rec.array(ar)

    def amplitudes(self, qubit):
        """Unique qubit amplitudes."""
        return np.unique(self[qubit].amp)

    def frequencies(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)


def _acquisition(
    params: RabiAmplitudeFrequencyParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeFreqData:
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

    data = RabiAmplitudeFreqData(durations=durations)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper_amp,
        sweeper_freq,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        prob = result.probability(state=1)
        data.register_qubit(
            qubit=qubit,
            freq=qd_pulses[qubit].frequency + frequency_range,
            amp=qd_pulses[qubit].amplitude * amplitude_range,
            prob=prob.tolist(),
            error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
        )
    return data


def _fit(data: RabiAmplitudeFreqData) -> RabiAmplitudeFrequencyResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_amplitudes = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in data.data:
        amps = data.amplitudes(qubit)
        freqs = data.frequencies(qubit)
        probability = data[qubit].prob
        probability_matrix = probability.reshape(len(amps), len(freqs)).T

        # guess optimal frequency maximizing oscillatio amplitude
        index = np.argmax([max(x) - min(x) for x in probability_matrix])
        frequency = freqs[index]

        y = probability_matrix[index, :].ravel()
        error = data[qubit].error.reshape(len(amps), len(freqs)).T
        error = error[index, :].ravel()

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(amps)
        x_max = np.max(amps)
        x = (amps - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=10)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = amps[index] / (amps[1] - amps[0]) if index is not None else 0.5

        pguess = [0.5, 0.5, 1 / f, 0]

        try:
            popt, perr = curve_fit(
                rabi_amplitude_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
                sigma=error,
            )
            perr = np.sqrt(np.diag(perr))
            pi_pulse_parameter = popt[2] / 2 * period_correction_factor(phase=popt[3])
            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = (pi_pulse_parameter, perr[2] / 2)
            fitted_parameters[qubit] = popt.tolist()

            chi2[qubit] = (
                chi2_reduced(
                    y,
                    rabi_amplitude_function(x, *popt),
                    error,
                ),
                np.sqrt(2 / len(y)),
            )
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeFrequencyResults(
        amplitude=fitted_amplitudes,
        length=data.durations,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        chi2=chi2,
    )


def _plot(
    data: RabiAmplitudeFreqData,
    target: QubitId,
    fit: RabiAmplitudeFrequencyResults = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=("Probability",),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp

    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=amplitudes,
                y=frequencies,
                z=qubit_data.prob,
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
        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", "Pi-pulse amplitude"],
                [
                    fit.frequency[target],
                    f"{fit.amplitude[target][0]:.6f} +- {fit.amplitude[target][1]:.6f} [a.u.]",
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend=dict(orientation="h"),
    )
    return figures, fitting_report


rabi_amplitude_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
