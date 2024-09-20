"""Rabi experiment that sweeps amplitude and frequency."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.identifier import QubitId
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine
from qibocal.config import log
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

from .amplitude_frequency_signal import (
    RabiAmplitudeFreqSignalData,
    RabiAmplitudeFrequencySignalParameters,
    RabiAmplitudeFrequencySignalResults,
    _update,
)
from .utils import fit_amplitude_function, rabi_amplitude_function, sequence_amplitude


@dataclass
class RabiAmplitudeFrequencyParameters(RabiAmplitudeFrequencySignalParameters):
    """RabiAmplitudeFrequency runcard inputs."""


@dataclass
class RabiAmplitudeFrequencyResults(RabiAmplitudeFrequencySignalResults):
    """RabiAmplitudeFrequency outputs."""

    chi2: dict[QubitId, list[float]] = field(default_factory=dict)


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
class RabiAmplitudeFreqData(RabiAmplitudeFreqSignalData):
    """RabiAmplitudeFreq data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiAmpFreqType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, prob, error):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        data = np.empty(size, dtype=RabiAmpFreqType)
        data["freq"] = frequency.ravel()
        data["amp"] = amplitude.ravel()
        data["prob"] = np.array(prob).ravel()
        data["error"] = np.array(error).ravel()
        self.data[qubit] = np.rec.array(data)


def _acquisition(
    params: RabiAmplitudeFrequencyParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeFreqData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    sequence, qd_pulses, ro_pulses, durations = sequence_amplitude(
        targets, params, platform
    )

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

        # guess optimal frequency maximizing oscillation amplitude
        index = np.argmax([max(x) - min(x) for x in probability_matrix])
        frequency = freqs[index]

        y = probability_matrix[index]
        error = data[qubit].error[data[qubit].freq == frequency]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(amps)
        x_max = np.max(amps)
        x = (amps - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        period = fallback_period(guess_period(amps, y))
        pguess = [0.5, 0.5, period, 0]

        try:
            popt, perr, pi_pulse_parameter = fit_amplitude_function(
                x,
                y,
                pguess,
                sigma=error,
                signal=False,
            )

            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = [pi_pulse_parameter, perr[2] / 2]
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
                y=[fit.frequency[target] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
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
        legend={"orientation": "h"},
    )
    return figures, fitting_report


rabi_amplitude_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
