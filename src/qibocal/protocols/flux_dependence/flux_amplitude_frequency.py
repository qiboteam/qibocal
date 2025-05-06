"""Experiment to compute detuning from flux pulses."""

import math
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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import HZ_TO_GHZ, table_dict, table_html

__all__ = ["flux_amplitude_frequency"]


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
    crosstalk_qubit: Optional[QubitId] = None
    """If provided a flux pulse will be applied on this qubit.

    Enable to compute the crosstalk matrix.

    """
    flux_pulse_amplitude: float = 0
    """Flux pulse amplitude on target qubits to bias from sweetstpot.

    It should be provided only if crosstalk is not None.
    """


@dataclass
class FluxAmplitudeFrequencyResults(Results):
    """FluxAmplitudeFrequency outputs."""

    crosstalk: bool = False
    """Check if this is crosstalk protocol."""
    detuning: dict[QubitId, float] = field(default_factory=dict)
    """Frequency detuning."""
    flux: dict[QubitId, float] = field(default_factory=dict)
    """Derived flux  """
    fitted_parameters_detuning: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted parameters for every qubit."""
    fitted_parameters_flux: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )

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

    crosstalk_qubit: Optional[QubitId]
    """Qubit where crosstalk will be measured."""
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
    detuning = {}
    for qubit in targets:
        if params.crosstalk_qubit is None and math.isclose(params.amplitude_min, 0):
            detuning[qubit] = 0
        else:
            assert (
                platform.calibration.single_qubits[qubit].qubit.flux_coefficients
                is not None
            ), (
                f"Flux coefficients for {qubit} missing. Re-run experiment starting with zero amplitude_min and without crosstalk qubit."
            )
            if params.crosstalk_qubit is not None:
                detuning[qubit] = platform.calibration.single_qubits[
                    qubit
                ].qubit.detuning(params.flux_pulse_amplitude)
            else:
                detuning[qubit] = platform.calibration.single_qubits[
                    qubit
                ].qubit.detuning(params.amplitude_min)

    qubit_frequency = {
        qubit: platform.calibration.single_qubits[qubit].qubit.frequency_01 * HZ_TO_GHZ
        for qubit in targets
    }

    data = FluxAmplitudeFrequencyData(
        crosstalk_qubit=params.crosstalk_qubit,
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
                target_amplitude=params.flux_pulse_amplitude,
                target_qubit=params.crosstalk_qubit,
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
                        (
                            params.crosstalk_qubit
                            if params.crosstalk_qubit is not None
                            else target
                        )
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
    crosstalk = data.crosstalk_qubit
    detuning = {}
    flux = {}
    qubits = np.unique([i[0] for i in data.data]).tolist()
    for qubit in qubits:
        amplitudes = data[qubit, "X"].amplitude
        X_exp = 2 * data[qubit, "X"].prob_1 - 1
        # TODO: check if sign of Y_exp is correct
        Y_exp = 1 - 2 * data[qubit, "Y"].prob_1
        phase = np.unwrap(np.angle(X_exp + 1j * Y_exp))
        # normalization required to avoid problems with arccos
        phase -= phase[0]
        other_det = data.detuning[qubit]
        f = data.qubit_frequency[qubit]
        det = phase / data.flux_pulse_duration / 2 / np.pi + other_det
        # to make sure that flux is invertible
        det[np.abs(det) < 1e-3] = 0
        # from inversion of flux dependence formula assuming negligible Ec and asymmetry
        derived_flux = 1 / np.pi * np.arccos(((f + det) / f) ** 2)
        flux[qubit] = derived_flux.tolist()
        fitted_parameters_detuning[qubit] = np.polyfit(amplitudes, det, 2).tolist()
        fitted_parameters_flux[qubit] = np.polyfit(amplitudes, derived_flux, 1).tolist()
        detuning[qubit] = det.tolist()
    return FluxAmplitudeFrequencyResults(
        crosstalk=crosstalk,
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
    fitting_report = ""
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
                name="Fit Detuning",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=amplitude,
                y=np.polyval(fit.fitted_parameters_flux[target], amplitude),
                name="Fit Flux",
            ),
            row=1,
            col=2,
        )
        if fit.crosstalk is None:
            fitting_report = table_html(
                table_dict(
                    target,
                    [
                        "Flux coefficients",
                        "Flux normalization",
                    ],
                    [
                        [
                            np.round(i, 3)
                            for i in fit.fitted_parameters_detuning[target]
                        ],
                        np.round(fit.fitted_parameters_flux[target][0], 3),
                    ],
                )
            )
        else:
            fitting_report = table_html(
                table_dict(
                    target,
                    [
                        f"Flux crosstalk with {fit.crosstalk}",
                    ],
                    [
                        np.round(fit.fitted_parameters_flux[target][0], 4),
                    ],
                )
            )

    fig.update_layout(
        showlegend=True,
        xaxis1_title="Flux pulse amplitude [a.u.]",
        xaxis2_title="Flux pulse amplitude [a.u.]",
        yaxis1_title="Detuning [GHz]",
        yaxis2_title="Flux [Flux quantum]",
    )

    return [fig], fitting_report


def _update(
    results: FluxAmplitudeFrequencyResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    if results.crosstalk is None:
        platform.calibration.single_qubits[
            target
        ].qubit.flux_coefficients = results.fitted_parameters_detuning[target]

    # TODO: needs to be inverted
    flux_qubit = results.crosstalk if results.crosstalk is not None else target
    update.crosstalk_matrix(
        results.fitted_parameters_flux[target][0], platform, target, flux_qubit
    )


flux_amplitude_frequency = Routine(_acquisition, _fit, _plot, _update)
