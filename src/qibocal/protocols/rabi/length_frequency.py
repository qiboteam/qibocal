"""Rabi experiment that sweeps length and frequency (with probability)."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ..utils import HZ_TO_GHZ, chi2_reduced, fallback_period, guess_period
from .length_frequency_signal import (
    RabiLengthFreqSignalData,
    RabiLengthFrequencySignalParameters,
    RabiLengthFrequencySignalResults,
    _update,
)
from .utils import fit_length_function, rabi_length_function, sequence_length


@dataclass
class RabiLengthFrequencyParameters(RabiLengthFrequencySignalParameters):
    """RabiLengthFrequency runcard inputs."""


@dataclass
class RabiLengthFrequencyResults(RabiLengthFrequencySignalResults):
    """RabiLengthFrequency outputs."""

    chi2: dict[QubitId, list[float]] = field(default_factory=dict)


RabiLenFreqType = np.dtype(
    [
        ("len", np.float64),
        ("freq", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqData(RabiLengthFreqSignalData):
    """RabiLengthFreq data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiLenFreqType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, lens, prob, error):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        frequency, length = np.meshgrid(freq, lens)
        data = np.empty(size, dtype=RabiLenFreqType)
        data["freq"] = frequency.ravel()
        data["len"] = length.ravel()
        data["prob"] = np.array(prob).ravel()
        data["error"] = np.array(error).ravel()
        self.data[qubit] = np.rec.array(data)


def _acquisition(
    params: RabiLengthFrequencyParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthFreqData:
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

    data = RabiLengthFreqData(amplitudes=amplitudes)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper_len,
        sweeper_freq,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        prob = result.probability(state=1)
        data.register_qubit(
            qubit=qubit,
            freq=qd_pulses[qubit].frequency + frequency_range,
            lens=length_range,
            prob=prob.tolist(),
            error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
        )
    return data


def _fit(data: RabiLengthFreqData) -> RabiLengthFrequencyResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_durations = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in data.data:
        durations = data.durations(qubit)
        freqs = data.frequencies(qubit)
        probability = data[qubit].prob
        probability_matrix = probability.reshape(len(durations), len(freqs)).T

        # guess optimal frequency maximizing oscillatio amplitude
        index = np.argmax([max(x) - min(x) for x in probability_matrix])
        frequency = freqs[index]

        y = probability_matrix[index, :].ravel()
        error = data[qubit].error[data[qubit].freq == frequency]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(durations)
        x_max = np.max(durations)
        x = (durations - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        period = fallback_period(guess_period(x, y))
        pguess = [0, np.sign(y[0]) * 0.5, period, 0, 0]

        try:
            popt, perr, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                sigma=error,
                signal=False,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            fitted_frequencies[qubit] = frequency
            fitted_durations[qubit] = [
                pi_pulse_parameter,
                perr[2] * (x_max - x_min) / 2,
            ]
            fitted_parameters[qubit] = popt
            chi2[qubit] = [
                chi2_reduced(
                    y,
                    rabi_length_function(x, *popt),
                    error,
                ),
                np.sqrt(2 / len(y)),
            ]

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthFrequencyResults(
        length=fitted_durations,
        amplitude=data.amplitudes,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        chi2=chi2,
    )


def _plot(
    data: RabiLengthFreqData,
    target: QubitId,
    fit: RabiLengthFrequencyResults = None,
):
    """Plotting function for RabiLengthFrequency."""
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
    durations = qubit_data.len

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=qubit_data.prob,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Durations [ns]", row=1, col=1)
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
        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", "Pi-pulse duration"],
                [
                    fit.frequency[target],
                    f"{fit.length[target][0]:.2f} +- {fit.length[target][1]:.2f} ns",
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )
    return figures, fitting_report


rabi_length_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Rabi length with frequency tuning."""
