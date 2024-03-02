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

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..utils import HZ_TO_GHZ


@dataclass
class RabiLengthFrequencySignalParameters(Parameters):
    """Rabi Length Frequency runcard inputs."""

    pulse_freq_width: int
    """Width [Hz] for frequency sweep, relative to the qubit frequency."""
    pulse_freq_step: int
    """Frequency step [Hz] for frequency sweep."""
    pulse_duration_start: float
    """Initial drive pulse duration [ns]."""
    pulse_duration_end: float
    """Final drive pulse duration [ns]."""
    pulse_duration_step: float
    """Drive pulse duration sweep step [ns]."""
    pulse_amplitude: Optional[float] = None
    """Drive pulse amplitude. Same for all qubits."""


@dataclass
class RabiLengthFrequencySignalResults(Results):
    """Rabi Length Frequency outputs."""

    frequency: dict[QubitId, float]
    """Pi pulse frequecy [GHz] for each qubit."""
    length: dict[QubitId, int]
    """Pi pulse duration [ns] for each qubit."""
    amplitude: dict[QubitId, float]
    """Pi pulse amplitude [a.u.]."""


RabiLengthFrequencySignalType = np.dtype(
    [("frequency", np.float64), ("length", np.float64), ("signal", np.float64)]
)
"""Custom dtype for Rabi Length Frequency."""


@dataclass
class RabiLengthFrequencySignalData(Data):
    """Rabi Length Frequency acquisition outputs."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Drive pulse amplitudes."""
    data: dict[QubitId, npt.NDArray[RabiLengthFrequencySignalType]] = field(
        default_factory=dict
    )
    """Raw signal data acquired."""

    def register_qubit(self, qubit, frequency, length, signal):
        """Store output for single qubit."""

        size = len(frequency) * len(length)
        ar = np.empty(size, dtype=RabiLengthFrequencySignalType)
        _frequency, _length = np.meshgrid(frequency, length)
        ar["frequency"] = _frequency.ravel()
        ar["length"] = _length.ravel()
        ar["signal"] = signal.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: RabiLengthFrequencySignalParameters, platform: Platform, qubits: Qubits
) -> RabiLengthFrequencySignalData:
    r"""
    Data acquisition for Rabi Length Frequency Experiment.
    In this variation of the Rabi experiment we not only sweep the pulse duration, but also its frequency. We aim to
    find the drive pulse length and frequency that creates a Pi rotation in X.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.pulse_duration_start
        )
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # qubit drive pulse frequency
    qd_pulse_frequency_range = np.arange(
        -params.pulse_freq_width // 2,
        params.pulse_freq_width // 2,
        params.pulse_freq_step,
    )
    frequency_sweeper = Sweeper(
        Parameter.frequency,
        qd_pulse_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    duration_sweeper = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    data = RabiLengthFrequencySignalData(amplitudes=amplitudes)

    # execute the sweep
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        duration_sweeper,
        frequency_sweeper,
    )

    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit,
            frequency=qd_pulse_frequency_range + qd_pulses[qubit].frequency,
            length=qd_pulse_duration_range,
            signal=result.magnitude,
        )
    return data


def _fit(data: RabiLengthFrequencySignalData) -> RabiLengthFrequencySignalResults:
    """Post-processing for Rabi Length Frequency experiment."""

    # qubits = data.qubits
    # fitted_parameters = {}
    # durations = {}

    # for qubit in qubits:
    #     qubit_data = data[qubit]
    #     rabi_parameter = qubit_data.length
    #     voltages = qubit_data.signal

    #     y_min = np.min(voltages)
    #     y_max = np.max(voltages)
    #     x_min = np.min(rabi_parameter)
    #     x_max = np.max(rabi_parameter)
    #     x = (rabi_parameter - x_min) / (x_max - x_min)
    #     y = (voltages - y_min) / (y_max - y_min) - 1 / 2

    #     # Guessing period using fourier transform
    #     ft = np.fft.rfft(y)
    #     mags = abs(ft)
    #     local_maxima = find_peaks(mags, threshold=1)[0]
    #     index = local_maxima[0] if len(local_maxima) > 0 else None
    #     # 0.5 hardcoded guess for less than one oscillation
    #     f = x[index] / (x[1] - x[0]) if index is not None else 0.5

    #     pguess = [0, np.sign(y[0]) * 0.5, 1 / f, 0, 0]
    #     try:
    #         popt, _ = curve_fit(
    #             utils.rabi_length_function,
    #             x,
    #             y,
    #             p0=pguess,
    #             maxfev=100000,
    #             bounds=(
    #                 [0, -1, 0, -np.pi, 0],
    #                 [1, 1, np.inf, np.pi, np.inf],
    #             ),
    #         )
    #         translated_popt = [  # change it according to the fit function
    #             (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
    #             (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
    #             popt[2] * (x_max - x_min),
    #             popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
    #             popt[4] / (x_max - x_min),
    #         ]
    #         pi_pulse_parameter = (
    #             translated_popt[2]
    #             / 2
    #             * utils.period_correction_factor(phase=translated_popt[3])
    #         )

    #     except:
    #         log.warning("rabi_fit: the fitting was not succesful")
    #         pi_pulse_parameter = 0
    #         translated_popt = [0, 0, 1, 0, 0]
    #     durations[qubit] = pi_pulse_parameter
    #     fitted_parameters[qubit] = translated_popt

    # return RabiLengthFrequencySignalResults(durations, data.amplitudes, fitted_parameters)
    return RabiLengthFrequencySignalResults({}, {}, {})


def _update(
    results: RabiLengthFrequencySignalResults, platform: Platform, qubit: QubitId
):
    # update.drive_duration(results.length[qubit], platform, qubit)
    # update.drive_frequency(results.frequency[qubit], platform, qubit)
    pass


def _plot(
    data: RabiLengthFrequencySignalData, fit: RabiLengthFrequencySignalResults, qubit
):
    """Plotting function for Rabi Length Frequency experiment."""
    figures = []
    fitting_report = ""

    qubit_data = data[qubit]

    subplot_titles = (
        "Signal [a.u.] Qubit" + str(qubit),
        "Spectral Density [a.u.] Qubit" + str(qubit),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Heatmap(
            x=qubit_data.frequency * HZ_TO_GHZ,
            y=qubit_data.length,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=f"Drive Pulse Frequency [GHz]",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Drive Pulse Duration [ns]", row=1, col=1)

    frequencies = np.unique(qubit_data.frequency)
    lengths = np.unique(qubit_data.length)
    signal = qubit_data.signal.reshape((len(lengths), len(frequencies)))
    signal -= np.mean(signal, axis=0)

    fft_magnitude = np.abs(np.fft.fft(signal, axis=0))[: len(lengths) // 2, :].ravel()
    fft_frequencies = np.repeat(
        np.fft.fftfreq(len(lengths), d=1 / 1000)[: len(lengths) // 2], len(frequencies)
    )

    fig.add_trace(
        go.Heatmap(
            x=qubit_data.frequency * HZ_TO_GHZ,
            y=fft_frequencies,
            z=fft_magnitude,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"Drive Pulse Frequency [GHz]",
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="FFT Frequency [MHz]", row=1, col=2)

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    figures.append(fig)

    return figures, fitting_report


rabi_frequency_length_signal = Routine(_acquisition, _fit, _plot, _update)
"""Rabi Length Frequency Routine object."""
