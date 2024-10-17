"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import Custom, FluxPulse, PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from ..utils import table_dict, table_html


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


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
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
        shape=Custom(np.ones(duration)),
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
                platform, qubit, duration, params
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

    fig = go.Figure()
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]

    X_exp = qubit_X_data.prob_1 - qubit_X_data.prob_0
    Y_exp = qubit_Y_data.prob_0 - qubit_Y_data.prob_1
    phase = np.unwrap(np.angle(X_exp + 1.0j * Y_exp))

    detuning = scipy.signal.savgol_filter(
        phase / 2 / np.pi,
        13,
        3,
        deriv=1,
        delta=1,
    )

    # TODO: understand why it works
    step_response_freq = detuning / np.average(
        detuning[-int(len(qubit_X_data.duration) / 2) :]
    )
    step_response_volt = np.sqrt(step_response_freq)

    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=step_response_volt,
            name="Volt response",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=np.ones(len(qubit_X_data.duration)),
            name="Ideal response",
        )
    )

    from scipy import optimize

    [A, tau], _ = optimize.curve_fit(
        exponential_decay,
        qubit_X_data.duration,
        step_response_volt,
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=exponential_decay(qubit_X_data.duration, A, tau),
            name="Fit",
        )
    )

    fir, iir = single_exponential_correction(A, tau)
    from scipy import signal

    no_filter = exponential_decay(qubit_X_data.duration, A, tau)
    step_response_th = np.ones(len(qubit_X_data))
    with_filter = no_filter * signal.lfilter(fir, [1, -iir[0]], step_response_th)

    fig.add_trace(
        go.Scatter(x=qubit_X_data.duration, y=with_filter, name="With filter"),
    )

    fig.update_layout(
        xaxis_title="Flux pulse duration [ns]",
        yaxis_title="Waveform [a.u.]",
    )

    fitting_report = table_html(
        table_dict(
            target,
            ["FIR", "IIR"],
            [fir, iir],
            display_error=False,
        )
    )

    return [fig], fitting_report


cryoscope = Routine(_acquisition, _fit, _plot)
