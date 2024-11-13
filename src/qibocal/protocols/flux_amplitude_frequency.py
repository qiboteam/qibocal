"""FluxAmplitudeFrequency experiment, corrects distortions."""

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

# from ..ramsey.utils import fitting, ramsey_fit
# from ..utils import table_dict, table_html, GHZ_TO_HZ


@dataclass
class FluxAmplitudeFrequencyParameters(Parameters):
    """FluxAmplitudeFrequency runcard inputs."""

    amplitude_min: int
    """Minimum flux pulse amplitude."""
    amplitude_max: int
    """Maximum flux amplitude."""
    amplitude_step: int
    """Flux pulse amplitude step."""
    duration: float
    """Flux pulse duration."""


@dataclass
class FluxAmplitudeFrequencyResults(Results):
    """FluxAmplitudeFrequency outputs."""

    fitted_parameters: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted parameters for every qubit."""

    # TODO: to be fixed
    def __contains__(self, key):
        return True


FluxAmplitudeFrequencyType = np.dtype([("amplitude", float), ("prob_1", np.float64)])
"""Custom dtype for FluxAmplitudeFrequency."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    amplitude: float,
    duration: int,
):

    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[qubit].flux

    flux_pulse = Pulse(duration=duration, amplitude=amplitude, envelope=Rectangular())

    # create the sequences
    sequence_x, sequence_y = PulseSequence(), PulseSequence()

    sequence_x.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=ry90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=flux_pulse.duration)),
            (drive_channel, ry90),
            (
                ro_channel,
                Delay(duration=ry90.duration + flux_pulse.duration + ry90.duration),
            ),
            (ro_channel, ro_pulse),
        ]
    )

    sequence_y.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=rx90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=flux_pulse.duration)),
            (drive_channel, rx90),
            (
                ro_channel,
                Delay(duration=ry90.duration + flux_pulse.duration + rx90.duration),
            ),
            (ro_channel, ro_pulse),
        ]
    )
    return sequence_x, sequence_y


@dataclass
class FluxAmplitudeFrequencyData(Data):
    """FluxAmplitudeFrequency acquisition outputs."""

    flux_pulse_duration: float
    """Flux pulse amplitude."""
    data: dict[tuple[QubitId, str], npt.NDArray[FluxAmplitudeFrequencyType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: FluxAmplitudeFrequencyParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FluxAmplitudeFrequencyData:

    data = FluxAmplitudeFrequencyData(
        flux_pulse_duration=params.duration,
    )

    sequences_x = []
    sequences_y = []

    amplitude_range = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
    )

    for amplitude in amplitude_range:
        sequence_x = PulseSequence()
        sequence_y = PulseSequence()

        for qubit in targets:
            qubit_sequence_x, qubit_sequence_y = generate_sequences(
                platform,
                qubit,
                duration=params.duration,
                amplitude=amplitude,
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

    for ig, (amplitude, sequence) in enumerate(zip(amplitude_range, sequences_x)):
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results_x[ig][ro_pulse.id]
            data.register_qubit(
                FluxAmplitudeFrequencyType,
                (qubit, "MX"),
                dict(
                    amplitude=np.array([amplitude]),
                    prob_1=result,
                ),
            )

    for ig, (amplitude, sequence) in enumerate(zip(amplitude_range, sequences_y)):
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results_y[ig][ro_pulse.id]
            data.register_qubit(
                FluxAmplitudeFrequencyType,
                (qubit, "MY"),
                dict(
                    amplitude=np.array([amplitude]),
                    prob_1=result,
                ),
            )

    return data


def _fit(data: FluxAmplitudeFrequencyData) -> FluxAmplitudeFrequencyResults:

    fitted_parameters = {}
    qubits = np.unique([i[0] for i in data.data])

    for qubit in qubits:
        amplitudes = data[qubit, "MX"].amplitude
        X_exp = 2 * data[qubit, "MX"].prob_1 - 1
        Y_exp = 2 * data[qubit, "MY"].prob_1 - 1

        phase = np.unwrap(np.angle(X_exp + 1j * Y_exp))
        detuning = phase / data.flux_pulse_duration / 2 / np.pi

        fitted_parameters[qubit] = np.polyfit(amplitudes, detuning, 2).tolist()

    return FluxAmplitudeFrequencyResults(fitted_parameters=fitted_parameters)


def _plot(
    data: FluxAmplitudeFrequencyData,
    fit: FluxAmplitudeFrequencyResults,
    target: QubitId,
):
    """FluxAmplitudeFrequency plots."""

    fig = go.Figure()

    amplitude = data[(target, "MX")].amplitude
    X_exp = 2 * data[(target, "MX")].prob_1 - 1
    Y_exp = 2 * data[(target, "MY")].prob_1 - 1
    phase = np.unwrap(np.angle(X_exp + 1j * Y_exp))
    detuning = phase / 2 / np.pi / data.flux_pulse_duration

    fig.add_trace(
        go.Scatter(
            x=amplitude,
            y=detuning,
            name="Detuning",
        )
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=np.polyval(fit.fitted_parameters[target], amplitude),
                name="fit",
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Flux pulse amplitude [a.u.]",
        yaxis_title="Detuning [GHz]",
    )

    return [fig], ""


flux_amplitude_frequency = Routine(_acquisition, _fit, _plot)
