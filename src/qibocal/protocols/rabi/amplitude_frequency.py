"""Rabi experiment that sweeps amplitude and frequency."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    chi2_reduced,
    table_dict,
    table_html,
)
from qibocal.result import probability

from .amplitude_frequency_signal import (
    RabiAmplitudeFreqSignalData,
    RabiAmplitudeFrequencySignalParameters,
    RabiAmplitudeFrequencySignalResults,
    _update,
)
from .utils import (
    fit_amplitude_function,
    plot_probabilities,
    rabi_amplitude_function,
    rabi_initial_guess,
    sequence_amplitude,
)

__all__ = ["rabi_amplitude_frequency"]


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
    params: RabiAmplitudeFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiAmplitudeFreqData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    sequence, qd_pulses, ro_pulses, durations = sequence_amplitude(
        targets, params, platform, params.rx90
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
    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    data = RabiAmplitudeFreqData(durations=durations, rx90=params.rx90)

    results = platform.execute(
        [sequence],
        [[amp_sweeper], [freq_sweepers[q] for q in targets]],
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
            amp=amp_sweeper.values,
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
        # here prob_matrix has dimensions (n_freqs, n_amps), so we
        # need to compute initial guesses over axis==1
        full_pguesses = rabi_initial_guess(
            amps, probability_matrix, "amp", signal=False, axis=1
        )

        # guess has the following elements:
        # 0. median guess
        # 1. amplitude guess
        # 2. period guess
        # 3. phase guess
        # 4. decaying constant guess
        # we estimate the best frequency by maximizing the amplitude estimation
        index = np.argmax(full_pguesses[1])

        frequency = freqs[index]
        y = probability_matrix[index, :].ravel()
        error = data[qubit].error[data[qubit].freq == frequency]

        # initial guesses for the best frequency row
        pguess = [p[index] for p in full_pguesses]
        try:
            popt, perr, pi_pulse_parameter = fit_amplitude_function(
                amps,
                y,
                pguess,
                sigma=error,
            )
            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = [pi_pulse_parameter, perr[2] / 2]
            fitted_parameters[qubit] = popt if isinstance(popt, list) else popt.tolist()
            chi2[qubit] = (
                chi2_reduced(
                    y,
                    rabi_amplitude_function(amps, *popt),
                    error,
                ),
                np.sqrt(2 / len(y)),
            )
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeFrequencyResults(
        amplitude=fitted_amplitudes,
        length={key: [value, 0] for key, value in data.durations.items()},
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        chi2=chi2,
        rx90=data.rx90,
    )


def _plot(
    data: RabiAmplitudeFreqData,
    target: QubitId,
    fit: RabiAmplitudeFrequencyResults | None = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    frequencies = data.frequencies(target) * HZ_TO_GHZ
    amplitudes = data.amplitudes(target)
    qubit_data = data[target]

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=qubit_data.prob.reshape(len(amplitudes), len(frequencies)).T,
        ),
    )
    fig.update_layout(
        title="Probability",
        xaxis_title="Amplitude [a.u.]",
        yaxis_title="Frequency [GHz]",
    )

    if fit is not None:
        selected_frequency = fit.frequency[target]

        fig.add_trace(
            go.Scatter(
                x=[min(amplitudes), max(amplitudes)],
                y=[selected_frequency * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
        )
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", f"{pulse_name} amplitude"],
                [
                    fit.frequency[target],
                    f"{fit.amplitude[target][0]:.6f} +- {fit.amplitude[target][1]:.6f} [a.u.]",
                ],
            )
        )

        fitted_data = data.return_row_data(selected_frequency, target)
        rabi1d_figure, rabi1d_report = plot_probabilities(
            fitted_data, target, fit, data.rx90
        )
        fitting_report += rabi1d_report
        figures.extend(rabi1d_figure)

    figures.insert(0, fig)

    return figures, fitting_report


rabi_amplitude_frequency = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
