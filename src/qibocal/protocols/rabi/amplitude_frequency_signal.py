"""Rabi experiment that sweeps amplitude and frequency (with probability)."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ParallelSweepers, Parameter, Sweeper

from qibocal.auto.operation import Protocol, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html
from qibocal.result import collect, magnitude, phase

from .acquisition import define_qubits_and_drivelines, sequence_amplitude
from .amplitude_frequency import RabiAmplitudeFreqClassificationData
from .parent_classes import (
    RabiAmplitudeFrequencyParameters,
    RabiFreqResults,
)
from .processing import (
    fit_amplitude_function,
    rabi_initial_guess,
    update_rabi_ampl_params,
)

__all__ = ["rabi_amplitude_frequency_signal"]


RabiAmpFreqSignalType = np.dtype(
    [
        ("amp", np.float64),
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeFreqSignalData(RabiAmplitudeFreqClassificationData):
    """RabiAmplitudeFreqSignal data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiAmpFreqSignalType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, i, q):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        data = np.empty(size, dtype=RabiAmpFreqSignalType)
        data["freq"] = frequency.ravel()
        data["amp"] = amplitude.ravel()
        data["i"] = np.array(i).ravel()
        data["q"] = np.array(q).ravel()
        self.data[qubit] = np.rec.array(data)

    def sig_mag(self, qubit):
        """comput signal from IQ components for a specific qubit."""
        return magnitude(collect(i=self[qubit].i, q=self[qubit].q))

    def sig_phase(self, qubit):
        """comput signal from IQ components for a specific qubit."""
        return phase(collect(i=self[qubit].i, q=self[qubit].q))


def _acquisition(
    params: RabiAmplitudeFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId] | list[QubitPairId],
) -> RabiAmplitudeFreqSignalData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    qubits_list, drive_lines = define_qubits_and_drivelines(targets)

    # create a sequence of pulses for the experiment
    sequence, qd_pulses, durations, updates = sequence_amplitude(
        targets=qubits_list,
        drive_lines=drive_lines,
        platform=platform,
        pulse_duration=params.pulse_length,
        pulse_ampl=None,  # in this case we are sweeping on amplitude
        rx90=params.rx90,
    )

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.amplitude_range,
        pulses=qd_pulses,
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

    data = RabiAmplitudeFreqSignalData(
        durations=durations,
        rx90=params.rx90,
    )

    updates |= {
        platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
        for q in qubits_list
    }
    results = platform.execute(
        [sequence],
        [
            ParallelSweepers([amp_sweeper]),
            ParallelSweepers([freq_sweepers[q] for q in qubits_list]),
        ],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in qubits_list:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        result = results[ro_pulse.id]
        data.register_qubit(
            qubit=qubit,
            freq=freq_sweepers[qubit].values,
            amp=amp_sweeper.values,
            i=result[..., 0],
            q=result[..., 1],
        )
    return data


def _fit(data: RabiAmplitudeFreqSignalData) -> RabiFreqResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_amplitudes = {}
    fitted_parameters = {}

    for qubit in data.data:
        amps = data.amplitudes_arr(qubit)
        freqs = data.frequencies_arr(qubit)
        signal = data.sig_mag(qubit)
        signal_matrix = signal.reshape(len(amps), len(freqs)).T

        # guess optimal frequency maximizing oscillation amplitude
        index = np.argmax([max(x) - min(x) for x in signal_matrix])
        frequency = freqs[index]

        # Guessing period using fourier transform
        y = signal_matrix[index]

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(amps)
        x_max = np.max(amps)
        x = (amps - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        pguess = rabi_initial_guess(x, y, "amp", signal=True)

        try:
            popt, _, pi_pulse_parameter = fit_amplitude_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = [pi_pulse_parameter]
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiFreqResults(
        amplitude=fitted_amplitudes,
        length={k: [v] for k, v in data.durations.items()},
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        rx90=data.rx90,
    )


def _plot(
    data: RabiAmplitudeFreqSignalData,
    target: QubitId | QubitPairId,
    fit: RabiFreqResults | None = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    qubit, drive_line = target if isinstance(target, tuple) else (target, target)

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
    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=data.sig_mag(qubit),
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=data.sig_phase(qubit),
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    figures.append(fig)

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[min(amplitudes), max(amplitudes)],
                y=[fit.frequency[qubit] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[min(amplitudes), max(amplitudes)],
                y=[fit.frequency[qubit] * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
            row=1,
            col=2,
        )
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                qubit,
                ["Optimal rabi frequency", f"{pulse_name} amplitude"],
                [
                    fit.frequency[qubit],
                    f"{fit.amplitude[qubit]:.6f} [a.u]",
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
        title=(f"Rabi experiment for qubit {qubit} with " + f"drive line {drive_line}"),
    )
    return figures, fitting_report


def _update(
    results: RabiFreqResults,
    platform: CalibrationPlatform,
    target: QubitId | QubitPairId,
) -> None:
    return update_rabi_ampl_params(
        results=results,
        platform=platform,
        target=target,
        label="signal",
    )


rabi_amplitude_frequency_signal = Protocol(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Rabi amplitude with frequency tuning."""
