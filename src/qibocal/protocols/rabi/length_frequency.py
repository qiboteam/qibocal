"""Rabi experiment that sweeps length and frequency."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ParallelSweepers, Parameter, Sweeper

from qibocal.auto.operation import Protocol, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import HZ_TO_GHZ, chi2_reduced, table_dict, table_html

from .acquisition import define_qubits_and_drivelines, sequence_length
from .parent_classes import (
    RabiData,
    RabiFreqResults,
    RabiLengthFrequencyParameters,
)
from .processing import (
    fit_length_function,
    rabi_initial_guess,
    rabi_length_function,
    update_rabi_parameters,
)

__all__ = ["rabi_length_frequency"]


RabiLenFreqClassType = np.dtype(
    [
        ("len", np.float64),
        ("freq", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqClassificationData(RabiData):
    """RabiLengthFreq data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiLenFreqClassType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, lens, prob, error):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        frequency, length = np.meshgrid(freq, lens)
        data = np.empty(size, dtype=RabiLenFreqClassType)
        data["freq"] = frequency.ravel()
        data["len"] = length.ravel()
        data["prob"] = np.array(prob).ravel()
        data["error"] = np.array(error).ravel()
        self.data[qubit] = np.rec.array(data)

    def durations_arr(self, qubit):
        """Unique qubit lengths."""
        return np.unique(self[qubit].len)

    def frequencies_arr(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)


def _acquisition(
    params: RabiLengthFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId] | list[QubitPairId],
) -> RabiLengthFreqClassificationData:
    """Data acquisition for Rabi experiment sweeping length."""

    qubits_list, drive_lines = define_qubits_and_drivelines(targets)

    sequence, qd_pulses, delays, amplitudes, updates = sequence_length(
        targets=qubits_list,
        drive_lines=drive_lines,
        platform=platform,
        pulse_ampl=params.pulse_amplitude,
        pulse_duration=None,  # in this case we are sweeping on duration
        rx90=params.rx90,
        use_align=params.interpolated_sweeper,
    )

    if params.interpolated_sweeper:
        # in this case delays is always an empty list, so it is safe to sum to qd_pulses
        len_sweep_param = Parameter.duration_interpolated
    else:
        len_sweep_param = Parameter.duration

    len_sweeper = Sweeper(
        parameter=len_sweep_param,
        range=params.duration_range,
        pulses=qd_pulses + delays,
    )

    frequency_values = np.arange(*params.frequency_range)
    freq_sweepers = {}
    for qubit, drive in zip(qubits_list, drive_lines):
        channel = platform.qubits[drive].drive
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(channel).frequency + frequency_values,
            channels=[channel],
        )

    data = RabiLengthFreqClassificationData(
        rx90=params.rx90,
        amplitudes=amplitudes,
    )

    results = platform.execute(
        [sequence],
        [
            ParallelSweepers([len_sweeper]),
            ParallelSweepers([freq_sweepers[q] for q in qubits_list]),
        ],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in qubits_list:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        prob = results[ro_pulse.id]
        data.register_qubit(
            qubit=qubit,
            freq=freq_sweepers[qubit].values,
            lens=len_sweeper.values,
            prob=prob.tolist(),
            error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
        )
    return data


def _fit(data: RabiLengthFreqClassificationData) -> RabiFreqResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_durations = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in data.data:
        durations = data.durations_arr(qubit)
        freqs = data.frequencies_arr(qubit)
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

        pguess = rabi_initial_guess(x, y, "length", signal=False)

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

    return RabiFreqResults(
        length=fitted_durations,
        amplitude={k: [v] for k, v in data.amplitudes.items()},
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        chi2=chi2,
        rx90=data.rx90,
    )


def _plot(
    data: RabiLengthFreqClassificationData,
    target: QubitId | QubitPairId,
    fit: RabiFreqResults | None = None,
):
    """Plotting function for RabiLengthFrequency."""
    qubit, drive_line = target if isinstance(target, tuple) else (target, target)

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=("Probability",),
    )
    qubit_data = data[qubit]
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

    fig.update_xaxes(title_text="Duration [ns]", row=1, col=1)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[min(durations), max(durations)],
                y=[fit.frequency[qubit] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                qubit,
                ["Optimal rabi frequency", f"{pulse_name} duration"],
                [
                    fit.frequency[qubit],
                    f"{fit.length[qubit][0]:.2f} +- {fit.length[qubit][1]:.2f} ns",
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
        title=(f"Rabi experiment for qubit {qubit} with " + f"drive line {drive_line}"),
    )
    return figures, fitting_report


rabi_length_frequency = Protocol(_acquisition, _fit, _plot, update_rabi_parameters)
"""Rabi length with frequency tuning."""
