"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
import scipy.signal
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Platform,
    Pulse,
    PulseSequence,
    Rectangular,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine

from ..ramsey.utils import fitting, ramsey_fit


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    padding: int = 0


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    fitted_parameters: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted parameters for every qubit."""

    # TODO: to be fixed
    def __contains__(self, key):
        return True


CryoscopeType = np.dtype([("duration", int), ("prob_1", np.float64)])
"""Custom dtype for Cryoscope."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    duration: int,
    params: CryoscopeParameters,
):

    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[qubit].flux

    flux_pulse = Pulse(
        duration=duration, amplitude=params.flux_pulse_amplitude, envelope=Rectangular()
    )

    # create the sequences
    sequence_x, sequence_y = PulseSequence(), PulseSequence()

    sequence_x.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=ry90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=params.duration_max + 100)),
            (drive_channel, ry90),
            (
                ro_channel,
                Delay(
                    duration=ry90.duration + params.duration_max + 100 + ry90.duration
                ),
            ),
            (ro_channel, ro_pulse),
        ]
    )

    sequence_y.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=rx90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=params.duration_max + 100)),
            (drive_channel, rx90),
            (
                ro_channel,
                Delay(
                    duration=ry90.duration + params.duration_max + 100 + rx90.duration
                ),
            ),
            (ro_channel, ro_pulse),
        ]
    )
    return sequence_x, sequence_y


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryoscopeData:

    data = CryoscopeData(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    sequences_x = []
    sequences_y = []

    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    for duration in duration_range:
        sequence_x = PulseSequence()
        sequence_y = PulseSequence()

        for qubit in targets:
            qubit_sequence_x, qubit_sequence_y = generate_sequences(
                platform, qubit, duration, params
            )
            sequence_x += qubit_sequence_x
            sequence_y += qubit_sequence_y

        sequences_x.append(sequence_x)
        sequences_y.append(sequence_y)

    options = dict(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    results_x = [platform.execute([sequence], **options) for sequence in sequences_x]
    results_y = [platform.execute([sequence], **options) for sequence in sequences_y]

    for ig, (duration, sequence) in enumerate(zip(duration_range, sequences_x)):
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results_x[ig][ro_pulse.id]
            data.register_qubit(
                CryoscopeType,
                (qubit, "MX"),
                dict(
                    duration=np.array([duration]),
                    prob_1=result,
                ),
            )

    for ig, (duration, sequence) in enumerate(zip(duration_range, sequences_y)):
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results_y[ig][ro_pulse.id]
            data.register_qubit(
                CryoscopeType,
                (qubit, "MY"),
                dict(
                    duration=np.array([duration]),
                    prob_1=result,
                ),
            )

    return data


def _fit(data: CryoscopeData) -> CryoscopeResults:

    fitted_parameters = {}
    for qubit, setup in data.data:
        qubit_data = data[qubit, setup]
        x = qubit_data.duration
        y = 2 * qubit_data.prob_1 - 1

        popt, _ = fitting(x, y)

        fitted_parameters[qubit, setup] = popt

    return CryoscopeResults(fitted_parameters=fitted_parameters)


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""

    fig = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    duration = data[(target, "MX")].duration
    X_exp = 2 * data[(target, "MX")].prob_1 - 1
    Y_exp = 2 * data[(target, "MY")].prob_1 - 1

    fig.add_trace(
        go.Scatter(
            x=duration,
            y=X_exp,
            name="X",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=duration,
            y=Y_exp,
            name="Y",
        ),
        row=1,
        col=1,
    )

    if fit is not None:
        x = np.linspace(np.min(duration), np.max(duration), 100)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ramsey_fit(x, *fit.fitted_parameters[target, "MX"]),
                name="Fit X",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ramsey_fit(x, *fit.fitted_parameters[target, "MY"]),
                name="Fit Y",
            ),
            row=1,
            col=1,
        )
    # detuning = np.unwrap(np.arctan2(qubit_Y_data.prob_1, qubit_X_data.prob_1))

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=detuning,
    #         name="detuning",
    #     ),
    #     row=2,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=[fit.fitted_parameters[target, "MY"][2]]*len(qubit_X_data.duration),
    #         name="expected detuning",
    #     ),
    #     row=2,
    #     col=1,
    # )

    # X_exp = 2*qubit_X_data.prob_1 - 1
    # Y_exp = 2*qubit_Y_data.prob_1 - 1

    # phase = np.angle(X_exp + 1j*Y_exp)
    # unwrap = np.unwrap(phase)

    # def normalize_sincos(
    #     data,
    #     window_size_frac=500,
    #     window_size=None,
    #     do_envelope=True):

    #     if window_size is None:
    #         window_size = len(data) // window_size_frac

    #         # window size for savgol filter must be odd
    #         window_size -= (window_size + 1) % 2
    #     mean_data_r = scipy.signal.savgol_filter(data.real, window_size, 0, 0)
    #     mean_data_i = scipy.signal.savgol_filter(data.imag, window_size, 0, 0)

    #     mean_data = mean_data_r + 1j * mean_data_i

    #     if do_envelope:
    #         envelope = np.sqrt(
    #             scipy.signal.savgol_filter(
    #                 (np.abs(
    #                     data -
    #                     mean_data))**2,
    #                 window_size,
    #                 0,
    #                 0))
    #     else:
    #         envelope = 1
    #     norm_data = ((data - mean_data) / envelope)
    #     return norm_data

    norm_data = X_exp + 1j * Y_exp

    # def fft_based_freq_guess_complex(y):
    #     """
    #     guess the shape of a sinusoidal complex signal y (in multiples of
    #         sampling rate), by selecting the peak in the fft.
    #     return guess (f, ph, off, amp) for the model
    #         y = amp*exp(2pi i f t + ph) + off.
    #     """
    #     fft = np.fft.fft(y)[1:len(y)]
    #     freq_guess_idx = np.argmax(np.abs(fft))
    #     if freq_guess_idx >= len(y) // 2:
    #         freq_guess_idx -= len(y)
    #     freq_guess = 1 / len(y) * (freq_guess_idx + 1)

    #     phase_guess = np.angle(fft[freq_guess_idx]) + np.pi / 2
    #     amp_guess = np.absolute(fft[freq_guess_idx]) / len(y)
    #     offset_guess = np.mean(y)

    #     return freq_guess, phase_guess, offset_guess, amp_guess

    # sampling_rate = 1
    # demod_freq = - \
    #             fft_based_freq_guess_complex(norm_data)[
    #                 0] * sampling_rate
    # print("DEMOD FREQ", demod_freq)
    # demod_data = np.exp(
    #         2 * np.pi * 1j * duration * demod_freq) * (norm_data)
    # phase = np.unwrap(np.angle(demod_data))
    # fig.add_trace(
    #     go.Scatter(
    #         x=duration,
    #         y=phase,
    #         name="detuning1",
    #     ),
    #     row=2,
    #     col=1,
    # )
    # print(demod_freq)
    # print(fit.fitted_parameters[target, "MY"][2])
    # print(fit.fitted_parameters[target, "MY"][2]/2/np.pi)

    demod_freq = -fit.fitted_parameters[target, "MY"][2] / 2 / np.pi
    demod_data = np.exp(2 * np.pi * 1j * duration * demod_freq) * (norm_data)
    phase = np.unwrap(np.angle(demod_data))
    detuning = scipy.signal.savgol_filter(
        phase / (2 * np.pi), window_length=7, polyorder=2, deriv=1
    )

    # def get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order):

    #     real_detuning = detuning-demod_freq+sampling_rate*nyquist_order
    #     return real_detuning

    # real_detuning = get_real_detuning(detuning, demod_freq, 1, 0)
    fig.add_trace(
        go.Scatter(
            x=duration,
            y=detuning,
            name="detuning",
        ),
        row=2,
        col=1,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=real_detuning,
    #         name="real detuning",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=Y_exp,
    #         name="Y_EXP",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=norm_data.real,
    #         name="X",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=norm_data.imag,
    #         name="Y",
    #     )
    # )
    return [fig], ""


cryoscope = Routine(_acquisition, _fit, _plot)
