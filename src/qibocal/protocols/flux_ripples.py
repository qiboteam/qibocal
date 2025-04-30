"""FluxGate experiment, implementation of Z gate using flux pulse."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Platform,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine

from ..result import probability
from .utils import COLORBAND, COLORBAND_LINE


@dataclass
class FluxGateParameters(Parameters):
    """FluxGate runcard inputs."""

    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    flux_pulse_duration: float
    drive_delay: int = 1


@dataclass
class FluxGateResults(Results):
    """FluxGate outputs."""


FluxGateType = np.dtype(
    [("duration", int), ("prob_1", np.float64), ("error", np.float64)]
)
"""Custom dtype for FluxGate."""


@dataclass
class FluxGateData(Data):
    """FluxGate acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    data: dict[tuple[QubitId, str], npt.NDArray[FluxGateType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: FluxGateParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FluxGateData:
    data = FluxGateData(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    sequence = PulseSequence()

    flux_zeros = {}
    # delays = []
    for qubit in targets:
        qubit_sequence = PulseSequence()
        native = platform.natives.single_qubit[qubit]

        drive_channel, rx90 = native.R(theta=np.pi / 2)[0]
        _, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
        ro_channel, ro_pulse = native.MZ()[0]
        flux_channel = platform.qubits[qubit].flux
        flux_pulse = Pulse(
            duration=params.flux_pulse_duration,
            amplitude=params.flux_pulse_amplitude,
            envelope=Rectangular(),
        )
        flux_zeros[qubit] = Pulse(
            duration=params.duration_max,
            amplitude=0,
            envelope=Rectangular(),
        )
        drive_delay = Delay(duration=params.drive_delay)
        qubit_sequence.extend(
            [
                (flux_channel, flux_pulse),
                (flux_channel, flux_zeros[qubit]),
            ]
        )
        qubit_sequence.align([drive_channel, flux_channel])
        qubit_sequence.extend(
            [
                (drive_channel, rx90),
                (drive_channel, drive_delay),
                (drive_channel, ry90),
            ]
        )
        qubit_sequence.align([drive_channel, flux_channel, ro_channel])
        qubit_sequence.extend(
            [
                (ro_channel, ro_pulse),
            ]
        )
        # delays += [drive_delay]
        sequence += qubit_sequence
    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=duration_range,
        pulses=list(flux_zeros.values()),
    )

    options = dict(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    results = platform.execute([sequence], [[sweeper]], **options)

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        prob = probability(results[ro_pulse.id], state=1)
        data.register_qubit(
            FluxGateType,
            (qubit),
            dict(
                duration=duration_range,
                prob_1=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots),
            ),
        )

    return data


def _fit(data: FluxGateData) -> FluxGateResults:
    return FluxGateResults()


def _plot(data: FluxGateData, fit: FluxGateResults, target: QubitId):
    """FluxGate plots."""

    fig = go.Figure()
    fitting_report = ""
    qubit_data = data[target]
    duration = qubit_data.duration
    prob = qubit_data.prob_1
    error = qubit_data.error
    fig.add_trace(
        go.Scatter(
            x=qubit_data.duration,
            y=qubit_data.prob_1,
            name="Data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate((duration, duration[::-1])),
            y=np.concatenate((prob + error, (prob - error)[::-1])),
            fill="toself",
            fillcolor=COLORBAND,
            line=dict(color=COLORBAND_LINE),
            showlegend=True,
            name="Errors",
        )
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    return [fig], fitting_report


flux_ripples = Routine(_acquisition, _fit, _plot)
