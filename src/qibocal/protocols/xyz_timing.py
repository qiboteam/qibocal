"""XYZTiming experiment, implementation of Z gate using flux pulse."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
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

INITIAL_DELAY = 100
"""Initial delay for both drive and flux channels."""


@dataclass
class XYZTimingParameters(Parameters):
    """XYZTiming runcard inputs."""

    delay_min: int
    """Minimum flux pulse duration."""
    delay_max: int
    """Maximum flux duration start."""
    delay_step: int
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    flux_pulse_duration: int
    """Flux pulse amplitude."""


@dataclass
class XYZTimingResults(Results):
    """XYZTiming outputs."""

    detuning: dict[QubitId, float] = field(default_factory=dict)
    """Detuning for every qubit."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Fitted parameters for every qubit."""


XYZTimingType = np.dtype([("duration", int), ("prob_1", np.float64)])
"""Custom dtype for XYZTiming."""


@dataclass
class XYZTimingData(Data):
    """XYZTiming acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    data: dict[tuple[QubitId, str], npt.NDArray[XYZTimingType]] = field(
        default_factory=dict
    )


def xyz_sequence(
    params: XYZTimingParameters, platform: Platform, delay: int, qubit: QubitId
) -> tuple[PulseSequence,]:

    qubit_sequence = PulseSequence()
    native = platform.natives.single_qubit[qubit]
    drive_channel, rx_pulse = native.RX()[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[qubit].flux
    flux_pulse = Pulse(
        duration=params.flux_pulse_duration,
        amplitude=params.flux_pulse_amplitude,
        envelope=Rectangular(),
    )

    my_delay = int(delay - (params.flux_pulse_duration - rx_pulse.duration) / 2)
    assert params.delay_min + 100 > 0, "Minimum delay allowed is 100."
    qubit_sequence.extend(
        [
            (drive_channel, Delay(duration=INITIAL_DELAY)),
            (flux_channel, Delay(duration=INITIAL_DELAY + my_delay)),
            (drive_channel, rx_pulse),
            (flux_channel, flux_pulse),
            (
                ro_channel,
                Delay(duration=INITIAL_DELAY + my_delay + flux_pulse.duration),
            ),
            (ro_channel, ro_pulse),
        ]
    )
    return qubit_sequence


def _acquisition(
    params: XYZTimingParameters,
    platform: Platform,
    targets: list[QubitId],
) -> XYZTimingData:

    data = XYZTimingData(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    delay_range = np.arange(params.delay_min, params.delay_max, params.delay_step)

    options = dict(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    sequences = []
    for delay in delay_range:
        seq = PulseSequence()
        for qubit in targets:
            seq += xyz_sequence(params, platform, delay, qubit)
        sequences.append(seq)

    results = [platform.execute([sequence], **options) for sequence in sequences]

    for i, sequence in enumerate(sequences):
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results[i]
            data.register_qubit(
                XYZTimingType,
                (qubit),
                dict(
                    duration=np.array([delay_range[i]]),
                    prob_1=result[ro_pulse.id],
                ),
            )

    return data


def _fit(data: XYZTimingData) -> XYZTimingResults:

    return XYZTimingResults()


def _plot(data: XYZTimingData, fit: XYZTimingResults, target: QubitId):
    """XYZTiming plots."""

    fig = go.Figure()
    fitting_report = ""
    qubit_data = data[target]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.duration,
            y=qubit_data.prob_1,
            name="1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_data.duration,
            y=1 - qubit_data.prob_1,
            name="0",
        )
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis=dict(range=[0, 1]),
        yaxis_title="Excited state probability",
    )

    return [fig], fitting_report


xyz_timing = Routine(_acquisition, _fit, _plot)
