"""Experiment to compute detuning from flux pulses."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from qibocal.calibration import CalibrationPlatform


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
    flux_qubit: Optional[QubitId] = None
    """For measuring flux crosstalk."""
    target_qubit_flux_amplitude: float = 0
    """Flux pulse amplitude to bias away qubit from sweetstpot."""


@dataclass
class FluxAmplitudeFrequencyResults(Results):
    """FluxAmplitudeFrequency outputs."""

    detuning: dict[QubitId, float] = field(default_factory=dict)
    """Frequency detuning."""
    flux: dict[QubitId, float] = field(default_factory=dict)
    """Frequency detuning."""
    fitted_parameters_detuning: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted parameters for every qubit."""
    fitted_parameters_flux: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )

    # TODO: to be fixed
    def __contains__(self, key):
        return True


FluxAmplitudeFrequencyType = np.dtype([("amplitude", float), ("prob_1", np.float64)])
"""Custom dtype for FluxAmplitudeFrequency."""


def ramsey_flux(
    platform: Platform,
    qubit: QubitId,
    amplitude: float,
    duration: int,
    measure: str,
    target_qubit: QubitId,
    target_amplitude: float,
):
    """Compute sequences at fixed amplitude of flux pulse for <X> and <Y>"""

    assert measure in ["X", "Y"]

    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[
        target_qubit if target_qubit is not None else qubit
    ].flux

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

    if target_qubit is not None:
        flux_channel = platform.qubits[qubit].flux
        flux_pulse = Pulse(
            duration=duration, amplitude=target_amplitude, envelope=Rectangular()
        )
        sequence.extend(
            [
                (flux_channel, Delay(duration=ry90.duration)),
                (flux_channel, flux_pulse),
            ]
        )
    return sequence


@dataclass
class FluxAmplitudeFrequencyData(Data):
    """FluxAmplitudeFrequency acquisition outputs."""

    flux_pulse_duration: float
    """Flux pulse amplitude."""
    qubit_frequency: dict = field(default_factory=dict)
    """Frequency of the qubits."""
    detuning: dict = field(default_factory=dict)
    """Detuning of the qubits."""
    data: dict[tuple[QubitId, str], npt.NDArray[FluxAmplitudeFrequencyType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: FluxAmplitudeFrequencyParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FluxAmplitudeFrequencyData:

    detuning = {
        qubit: (
            0
            if params.flux_qubit is None
            else platform.calibration.single_qubits[qubit].qubit.detuning(
                params.target_qubit_flux_amplitude
            )
        )
        for qubit in targets
    }
    qubit_frequency = {
        qubit: platform.calibration.single_qubits[qubit].qubit.frequency_01 / 1e9
        for qubit in targets
    }

    data = FluxAmplitudeFrequencyData(
        flux_pulse_duration=params.duration,
        qubit_frequency=qubit_frequency,
        detuning=detuning,
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
                target_amplitude=params.target_qubit_flux_amplitude,
                target_qubit=params.flux_qubit,
            )

        sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[
                pulse[1]
                for pulse in sequence
                if pulse[0]
                in [
                    platform.qubits[
                        params.flux_qubit if params.flux_qubit is not None else target
                    ].flux
                    for target in targets
                ]
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

    fitted_parameters_detuning = {}
    fitted_parameters_flux = {}
    detuning = {}
    flux = {}
    qubits = np.unique([i[0] for i in data.data]).tolist()
    for qubit in qubits:
        amplitudes = data[qubit, "X"].amplitude
        X_exp = 1 - 2 * data[qubit, "X"].prob_1
        Y_exp = 1 - 2 * data[qubit, "Y"].prob_1
        phase = np.unwrap(np.angle(X_exp + 1j * Y_exp))
        other_det = data.detuning[qubit]
        f = data.qubit_frequency[qubit]
        det = -phase / data.flux_pulse_duration / 2 / np.pi + other_det
        derived_flux = 1 / np.pi * np.arccos(((f + det) / f) ** 2)
        flux[qubit] = derived_flux.tolist()
        fitted_parameters_detuning[qubit] = np.polyfit(amplitudes, det, 2).tolist()
        fitted_parameters_flux[qubit] = np.polyfit(amplitudes, derived_flux, 1).tolist()
        detuning[qubit] = det.tolist()
    return FluxAmplitudeFrequencyResults(
        detuning=detuning,
        fitted_parameters_detuning=fitted_parameters_detuning,
        flux=flux,
        fitted_parameters_flux=fitted_parameters_flux,
    )


def _plot(
    data: FluxAmplitudeFrequencyData,
    fit: FluxAmplitudeFrequencyResults,
    target: QubitId,
):
    """FluxAmplitudeFrequency plots."""

    fig = make_subplots(
        rows=1,
        cols=2,
    )

    amplitude = data[(target, "X")].amplitude

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=fit.detuning[target],
                name="Detuning",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=fit.flux[target],
                name="Flux",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=np.polyval(fit.fitted_parameters_detuning[target], amplitude),
                name="fit",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=np.polyval(fit.fitted_parameters_flux[target], amplitude),
                name="fit",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        showlegend=True,
        xaxis1_title="Flux pulse amplitude [a.u.]",
        xaxis2_title="Flux pulse amplitude [a.u.]",
        yaxis1_title="Detuning [GHz]",
        yaxis2_title="Flux [Phi0]",
    )

    return [fig], ""


def _update(
    results: FluxAmplitudeFrequencyResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    pass


flux_amplitude_frequency = Routine(_acquisition, _fit, _plot)
