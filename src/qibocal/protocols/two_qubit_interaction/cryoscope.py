"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import Custom, FluxPulse, PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from .filters import exponential_decay, single_exponential_correction


def exponential_decay(x, a, t):
    return 1 + a * np.exp(-x / t)


def single_exponential_correction(
    A: float,
    tau: float,
):
    """
    Calculate the best FIR and IIR filter taps to correct for an exponential decay
    (undershoot or overshoot) of the shape
    `1 + A * exp(-t/tau)`.
    Args:
        A: The exponential decay pre-factor.
        tau: The time constant for the exponential decay, given in ns.
    Returns:
        A tuple of two items.
        The first is a list of 2 FIR (feedforward) taps starting at 0 and spaced `Ts` apart.
        The second is a single IIR (feedback) tap.
    """
    tau *= 1e-9
    Ts = 1e-9  # sampling rate
    k1 = Ts + 2 * tau * (A + 1)
    k2 = Ts - 2 * tau * (A + 1)
    c1 = Ts + 2 * tau
    c2 = Ts - 2 * tau
    feedback_tap = [-k2 / k1]
    feedforward_taps = list(np.array([c1, c2]) / k1)
    return feedforward_taps, feedback_tap


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
    parabola_coefficients: list[float]
    """Coefficients computed using FluxAmplitudeDetuning."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    padding: int = 0


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    pass


CryoscopeType = np.dtype(
    [("duration", int), ("prob_0", np.float64), ("prob_1", np.float64)]
)
"""Custom dtype for Cryoscope."""


def generate_waveform():
    # zeros = np.zeros(20)
    ones = np.ones(60)

    return np.concatenate([ones])


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    waveform: np.ndarray,
    duration: int,
    params: CryoscopeParameters,
):

    ry90 = platform.create_RX90_pulse(
        qubit,
        start=0,
        relative_phase=np.pi / 2,
    )

    # apply a detuning flux pulse
    flux_pulse = FluxPulse(
        start=ry90.finish + params.padding,
        duration=duration,
        amplitude=params.flux_pulse_amplitude,
        shape=Custom(waveform[:duration]),
        channel=platform.qubits[qubit].flux.name,
        qubit=qubit,
    )

    rx90 = platform.create_RX90_pulse(
        qubit,
        start=ry90.finish + params.duration_max + params.padding,
    )

    ry90_second = platform.create_RX90_pulse(
        qubit,
        start=ry90.finish + params.duration_max + params.padding,
        relative_phase=np.pi / 2,
    )

    ro = platform.create_qubit_readout_pulse(
        qubit, start=max(rx90.finish, ry90_second.finish)
    )

    # create the sequences
    sequence_x = PulseSequence(
        ry90,
        flux_pulse,
        ry90_second,
        ro,
    )

    sequence_y = PulseSequence(
        ry90,
        flux_pulse,
        rx90,
        ro,
    )
    return sequence_x, sequence_y


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    waveform: list
    """Flux pulse waveform"""
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryoscopeData:
    # define sequences of pulses to be executed
    data = CryoscopeData(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
        waveform=generate_waveform().tolist(),
    )

    sequences_x = []
    sequences_x_ro_pulses = []
    sequences_y = []
    sequences_y_ro_pulses = []

    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    for duration in duration_range:
        sequence_x = PulseSequence()
        sequence_y = PulseSequence()

        for qubit in targets:
            qubit_sequence_x, qubit_sequence_y = generate_sequences(
                platform, qubit, np.array(data.waveform), duration, params
            )
            sequence_x += qubit_sequence_x
            sequence_y += qubit_sequence_y

        sequences_x.append(sequence_x)
        sequences_y.append(sequence_y)
        sequences_x_ro_pulses.append(sequence_x.ro_pulses)
        sequences_y_ro_pulses.append(sequence_y.ro_pulses)

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    results_x = [
        platform.execute_pulse_sequence(sequence, options) for sequence in sequences_x
    ]
    results_y = [
        platform.execute_pulse_sequence(sequence, options) for sequence in sequences_y
    ]

    for ig, (duration, ro_pulses) in enumerate(
        zip(duration_range, sequences_x_ro_pulses)
    ):
        for qubit in targets:
            serial = ro_pulses.get_qubit_pulses(qubit)[0].serial
            result = results_x[ig][serial]
            data.register_qubit(
                CryoscopeType,
                (qubit, "MX"),
                dict(
                    duration=np.array([duration]),
                    prob_0=result.probability(state=0),
                    prob_1=result.probability(state=1),
                ),
            )
    for ig, (duration, ro_pulses) in enumerate(
        zip(duration_range, sequences_y_ro_pulses)
    ):
        for qubit in targets:
            serial = ro_pulses.get_qubit_pulses(qubit)[0].serial
        result = results_y[ig][serial]
        data.register_qubit(
            CryoscopeType,
            (qubit, "MY"),
            dict(
                duration=np.array([duration]),
                prob_0=result.probability(state=0),
                prob_1=result.probability(state=1),
            ),
        )

    return data


def _fit(data: CryoscopeData) -> CryoscopeResults:
    return CryoscopeResults()


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""
    figures = []
    fitting_report = f"Cryoscope of qubit {target}"

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(),
    )
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=qubit_X_data.prob_1,
    #         name="X",
    #         legendgroup="X",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_Y_data.duration,
    #         y=qubit_Y_data.prob_1,
    #         name="Y",
    #         legendgroup="Y",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # minus sign for X_exp becuase I get -cos phase
    X_exp = qubit_X_data.prob_1 - qubit_X_data.prob_0
    Y_exp = qubit_Y_data.prob_0 - qubit_Y_data.prob_1
    phase = np.unwrap(np.angle(X_exp + 1.0j * Y_exp))
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=phase,
    #         name="phase",
    #     ),
    #     row=2,
    #     col=1,
    # )

    # coeffs = [-9.10575082, -7.28208663e-3, -4.73157701e-5]  # D2
    coeffs = [[-9.09948820, 5.52686083e-3, -7.42079805e-5]]  # D2
    # coeffs = [-7.76584706, 2.25726809e-3, -3.76982885e-4]  # D1
    coeffs = [-7.76812980, 4.00605656e-02, -3.88473996e-4]  # D1
    coeffs = [-8.99178536, 5.19796241e-3, -1.61507231e-4]  # D3
    # coeffs = [-8.99334695, -6.70786688e-4, -2.15611619e-04] # D3
    # coeffs = [-9.10575082e+00, -7.28208663e-03, -4.73157701e-05]  # with filters
    detuning = scipy.signal.savgol_filter(
        phase / 2 / np.pi,
        13,
        3,
        deriv=1,
        delta=1,  # step
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=detuning,
    #         name="detuning",
    #     ),
    #     row=3,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=np.polyval(
    #             coeffs,
    #             (data.flux_pulse_amplitude) * np.ones(len(qubit_X_data.duration)),
    #         ),
    #         name="fit",
    #     ),
    #     row=3,
    #     col=1,
    # )
    step_response_freq = detuning / np.average(
        detuning[-int(len(qubit_X_data.duration) / 2) :]
    )
    step_response_volt = np.sqrt(step_response_freq)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=step_response_volt,
            name="Volt response",
        ),
        row=1,
        col=1,
    )
    print(step_response_volt)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=np.ones(len(qubit_X_data.duration)),
            name="Volt response",
        ),
        row=1,
        col=1,
    )

    from scipy import optimize

    [A, tau], _ = optimize.curve_fit(
        exponential_decay,
        qubit_X_data.duration,
        step_response_volt,
    )
    # xplot = qubit_X_data.duration - 20
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=exponential_decay(qubit_X_data.duration, A, tau),
            name="Fit",
        ),
        row=1,
        col=1,
    )

    print(A, tau)

    # fir, iir = filter_calc(exponential=[(A, tau)])

    fir, iir = single_exponential_correction(A, tau)
    print(f"FIR: {fir}\nIIR: {iir}")
    from scipy import signal

    no_filter = exponential_decay(qubit_X_data.duration, A, tau)
    step_response_th = np.ones(len(qubit_X_data))
    with_filter = no_filter * signal.lfilter(fir, [1, -iir[0]], step_response_th)

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration - 20,
    #         y=np.ones(len(qubit_X_data)),
    #         name="Theory",
    #     ),
    #     row=2,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x = xplot,
    #         y=no_filter,
    #         name="No filter"
    #     ),
    #     row=3,
    #     col=1,
    # )

    fig.add_trace(
        go.Scatter(x=qubit_X_data.duration, y=with_filter, name="With filter"),
        row=1,
        col=1,
    )

    # fig.add_trace(
    #     go.Scatter(
    #         x = xplot,
    #         y=np.ones(len(with_filter)),
    #         name="Theory"
    #     ),
    #     row=3,
    #     col=1,
    # )

    # detuning = np.abs(np.polyval(
    #             coeffs,
    #             (data.flux_pulse_amplitude) * np.array(data.waveform),
    #         ))
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.array(range(len(data.waveform))),
    #         y=(detuning - np.min(detuning) / (np.max(detuning) - np.min(detuning))),
    #         name="Waveform",
    #     ),
    #     row=4,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=qubit_X_data.duration,
    #         y=np.abs(scipy.signal.savgol_filter(
    #             (phase - phase[-1]) / 2 / np.pi,
    #             13,
    #             3,
    #             deriv=1,
    #         )),
    #         name="detuning",
    #     ),
    #     row=4,
    #     col=1,
    # )

    fig.update_layout(
        # xaxis3_title="Flux pulse duration [ns]",
        # yaxis1_title="Detuning [GHz]",
        # yaxis2_title="Waveform",
        # yaxis3_title="Detuning [GHz]",
    )
    return [fig], fitting_report


cryoscope = Routine(_acquisition, _fit, _plot)
