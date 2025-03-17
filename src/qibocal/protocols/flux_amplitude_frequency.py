"""Experiment to compute detuning from flux pulses."""

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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine


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

    detuning: dict[QubitId, float] = field(default_factory=dict)
    """Frequency detuning."""
    fitted_parameters: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted parameters for every qubit."""

    def __contains__(self, target: QubitId):
        return target in self.detuning


FluxAmplitudeFrequencyType = np.dtype([("amplitude", float), ("prob_1", np.float64)])
"""Custom dtype for FluxAmplitudeFrequency."""


def ramsey_flux(
    platform: Platform,
    qubit: QubitId,
    amplitude: float,
    duration: int,
    measure: str,
):
    """Compute sequences at fixed amplitude of flux pulse for <X> and <Y>"""

    assert measure in ["X", "Y"]

    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[qubit].flux

    flux_pulse = Pulse(duration=duration, amplitude=amplitude, envelope=Rectangular())

    # create the sequences
    sequence = PulseSequence()

    if measure == "X":
        sequence.extend(
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
    else:
        sequence.extend(
            [
                (drive_channel, ry90),
                (flux_channel, Delay(duration=ry90.duration)),
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
    return sequence


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
    amplitudes = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
    )

    options = dict(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for measure in ["X", "Y"]:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += ramsey_flux(
                platform,
                qubit,
                duration=params.duration,
                amplitude=params.amplitude_max / 2,
                measure=measure,
            )

        sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[
                pulse[1]
                for pulse in sequence
                if pulse[0] in [platform.qubits[target].flux for target in targets]
                and isinstance(pulse[1], Pulse)
            ],
        )
        result = platform.execute([sequence], [[sweeper]], **options)

        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            data.register_qubit(
                FluxAmplitudeFrequencyType,
                (qubit, measure),
                dict(
                    amplitude=amplitudes,
                    prob_1=result[ro_pulse.id],
                ),
            )

    return data


def _fit(data: FluxAmplitudeFrequencyData) -> FluxAmplitudeFrequencyResults:
    fitted_parameters = {}
    detuning = {}
    qubits = np.unique([i[0] for i in data.data]).tolist()

    for qubit in qubits:
        amplitudes = data[qubit, "X"].amplitude
        X_exp = 1 - 2 * data[qubit, "X"].prob_1
        Y_exp = 1 - 2 * data[qubit, "Y"].prob_1

        phase = np.unwrap(np.angle(X_exp + 1j * Y_exp))
        # normalize phase
        phase -= phase[0]
        det = phase / data.flux_pulse_duration / 2 / np.pi

        fitted_parameters[qubit] = np.polyfit(amplitudes, det, 2).tolist()
        detuning[qubit] = det.tolist()
    return FluxAmplitudeFrequencyResults(
        detuning=detuning, fitted_parameters=fitted_parameters
    )


def _plot(
    data: FluxAmplitudeFrequencyData,
    fit: FluxAmplitudeFrequencyResults,
    target: QubitId,
):
    """FluxAmplitudeFrequency plots."""

    fig = go.Figure()

    amplitude = data[(target, "X")].amplitude

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=fit.detuning[target],
                name="Detuning",
            )
        )
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


def _update(results: FluxAmplitudeFrequencyResults, platform: Platform, qubit: QubitId):
    update.flux_coefficients(results.fitted_parameters[qubit], platform, qubit)


flux_amplitude_frequency = Routine(_acquisition, _fit, _plot, _update)
