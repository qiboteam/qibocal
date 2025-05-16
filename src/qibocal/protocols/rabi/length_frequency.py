"""Rabi experiment that sweeps length and frequency (with probability)."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ...result import probability
from ..utils import HZ_TO_GHZ, chi2_reduced, fallback_period, guess_period
from .length_frequency_signal import (
    RabiLengthFreqSignalData,
    RabiLengthFrequencySignalParameters,
    RabiLengthFrequencySignalResults,
    _update,
)
from .utils import fit_length_function, rabi_length_function, sequence_length

__all__ = ["rabi_length_frequency"]


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
    params: RabiLengthFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthFreqData:
    """Data acquisition for Rabi experiment sweeping length."""

    sequence, qd_pulses, delays, ro_pulses, amplitudes = sequence_length(
        targets, params, platform, params.rx90
    )

    sweep_range = (
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    if params.interpolated_sweeper:
        len_sweeper = Sweeper(
            parameter=Parameter.duration_interpolated,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets],
        )
    else:
        len_sweeper = Sweeper(
            parameter=Parameter.duration,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets] + [delays[q] for q in targets],
        )

    frequency_range = np.arange(
        params.min_freq,
        params.max_freq,
        params.step_freq,
    )
    freq_sweepers = {}
    for qubit in targets:
        channel = platform.qubits[qubit].drive
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(channel).frequency + frequency_range,
            channels=[channel],
        )

    data = RabiLengthFreqData(amplitudes=amplitudes, rx90=params.rx90)

    results = platform.execute(
        [sequence],
        [[len_sweeper], [freq_sweepers[q] for q in targets]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        prob = probability(result, state=1)
        data.register_qubit(
            qubit=qubit,
            freq=freq_sweepers[qubit].values,
            lens=len_sweeper.values,
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
        rx90=data.rx90,
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
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", f"{pulse_name} duration"],
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
