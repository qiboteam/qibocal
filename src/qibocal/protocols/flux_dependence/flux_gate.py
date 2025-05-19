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

from ...result import probability
from ..ramsey.utils import fitting, ramsey_fit
from ..utils import COLORBAND, COLORBAND_LINE, GHZ_TO_HZ, table_dict, table_html

__all__ = ["flux_gate"]


@dataclass
class FluxGateParameters(Parameters):
    """FluxGate runcard inputs."""

    duration_min: float
    """Minimum flux pulse duration."""
    duration_max: float
    """Maximum flux duration start."""
    duration_step: float
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""


@dataclass
class FluxGateResults(Results):
    """FluxGate outputs."""

    detuning: dict[QubitId, float] = field(default_factory=dict)
    """Detuning for every qubit."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Fitted parameters for every qubit."""


FluxGateType = np.dtype(
    [("duration", float), ("prob_1", np.float64), ("error", np.float64)]
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

    flux_pulses = {}
    delays = []
    for qubit in targets:
        qubit_sequence = PulseSequence()
        native = platform.natives.single_qubit[qubit]

        drive_channel, rx90 = native.R(theta=np.pi / 2)[0]
        ro_channel, ro_pulse = native.MZ()[0]
        flux_channel = platform.qubits[qubit].flux
        flux_pulses[qubit] = Pulse(
            duration=params.duration_max,
            amplitude=params.flux_pulse_amplitude,
            envelope=Rectangular(),
        )
        drive_delay = Delay(duration=flux_pulses[qubit].duration)
        ro_delay = Delay(duration=flux_pulses[qubit].duration)
        qubit_sequence.extend(
            [
                (drive_channel, rx90),
                (flux_channel, Delay(duration=rx90.duration)),
                (flux_channel, flux_pulses[qubit]),
                (drive_channel, drive_delay),
                (drive_channel, rx90),
                (ro_channel, ro_delay),
                (ro_channel, Delay(duration=2 * rx90.duration)),
                (ro_channel, ro_pulse),
            ]
        )
        delays += [drive_delay, ro_delay]
        sequence += qubit_sequence
    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=duration_range,
        pulses=list(flux_pulses.values()) + delays,
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
    fitted_parameters = {}
    detuning = {}
    for qubit in data.qubits:
        qubit_data = data[qubit]
        x = qubit_data.duration
        y = qubit_data.prob_1

        popt, _ = fitting(x, y)
        fitted_parameters[qubit] = popt
        detuning[qubit] = popt[2] / (2 * np.pi) * GHZ_TO_HZ

    return FluxGateResults(detuning=detuning, fitted_parameters=fitted_parameters)


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

    if fit is not None:
        x = np.linspace(np.min(qubit_data.duration), np.max(qubit_data.duration), 100)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ramsey_fit(x, *fit.fitted_parameters[target]),
                name="Fit",
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Flux pulse amplitude", "Detuning [Hz]"],
                [data.flux_pulse_amplitude, fit.detuning[target]],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    return [fig], fitting_report


flux_gate = Routine(_acquisition, _fit, _plot)
