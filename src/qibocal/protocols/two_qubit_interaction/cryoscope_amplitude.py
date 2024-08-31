"""FluxAmplitudeDetuning experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class FluxAmplitudeDetuningParameters(Parameters):
    """FluxAmplitudeDetuning runcard inputs."""

    amplitude_min: float
    """Minimum flux pulse amplitude."""
    amplitude_max: float
    """Maximum flux pulse amplitude."""
    amplitude_step: float
    """Flux pulse amplitude step."""
    flux_pulse_amplitude: float
    """Flux pulse duration."""
    flux_pulse_duration: float
    """Flux pulse duration."""
    nshots: Optional[int] = None
    """Number of shots per point."""


@dataclass
class FluxAmplitudeDetuningResults(Results):
    """FluxAmplitudeDetuning outputs."""

    flux_coefficients: dict[QubitId, float] = field(default_factory=dict)


FluxAmplitudeDetuningType = np.dtype([("amplitude", float), ("prob", np.float64)])
"""Custom dtype for FluxAmplitudeDetuning."""


@dataclass
class FluxAmplitudeDetuningData(Data):
    """FluxAmplitudeDetuning acquisition outputs."""

    flux_pulse_duration: int
    data: dict[tuple[QubitId, str], npt.NDArray[FluxAmplitudeDetuningType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: FluxAmplitudeDetuningParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FluxAmplitudeDetuningData:

    sequence_x = PulseSequence()
    sequence_y = PulseSequence()

    initial_pulses = {}
    flux_pulses = {}
    rx90_pulses = {}
    ry90_pulses = {}
    ro_pulses = {}

    for qubit in targets:

        initial_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=0, relative_phase=np.pi / 2
        )
        flux_pulse_shape = Rectangular()
        flux_start = initial_pulses[qubit].finish
        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=flux_start,
            duration=params.flux_pulse_duration,
            amplitude=params.flux_pulse_amplitude,
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )
        # rotate around the X axis RX(-pi/2) to measure Y component
        rx90_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=initial_pulses[qubit].finish + flux_pulses[qubit].finish
        )
        # rotate around the Y axis RX(-pi/2) to measure X component
        ry90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_pulses[qubit].finish + flux_pulses[qubit].finish,
            relative_phase=np.pi / 2,
        )

        # add readout at the end of the sequences
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=rx90_pulses[qubit].finish  # to be fixed
        )

        # create the sequences
        sequence_x.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            ry90_pulses[qubit],  # rotate around Y to measure X CHECK
            ro_pulses[qubit],
        )
        sequence_y.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            rx90_pulses[qubit],  # rotate around X to measure Y CHECK
            ro_pulses[qubit],
        )
    amplitude_range = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
    )

    amp_sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.FACTOR,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = FluxAmplitudeDetuningData(flux_pulse_duration=params.flux_pulse_duration)

    for sequence, tag in [(sequence_x, "MX"), (sequence_y, "MY")]:
        results = platform.sweep(sequence, options, amp_sweeper)
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                FluxAmplitudeDetuningType,
                (qubit, tag),
                dict(
                    amplitude=amplitude_range * params.flux_pulse_amplitude,
                    prob=result.probability(state=1),
                ),
            )
    return data


def _fit(data: FluxAmplitudeDetuningData) -> FluxAmplitudeDetuningResults:

    return FluxAmplitudeDetuningResults()


def _plot(
    data: FluxAmplitudeDetuningData, fit: FluxAmplitudeDetuningResults, target: QubitId
):
    """FluxAmplitudeDetuning plots."""
    figures = []

    fitting_report = f"FluxAmplitudeDetuning of qubit {target}"

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            # f"Qubit {qubits[0]}",
            # f"Qubit {qubits[1]}",
        ),
    )
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=qubit_X_data.prob,
            name="X",
            legendgroup="X",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=qubit_Y_data.amplitude,
            y=qubit_Y_data.prob,
            name="Y",
            legendgroup="Y",
        ),
        row=1,
        col=1,
    )
    X_exp = 2 * qubit_X_data.prob - 1
    Y_exp = 1 - 2 * qubit_Y_data.prob
    phase = np.angle(X_exp + 1.0j * Y_exp)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=np.unwrap(phase),
            name="phase",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=np.unwrap(phase) / 2 / np.pi / data.flux_pulse_duration,
            name="Detuning [GHz]",
        ),
        row=3,
        col=1,
    )

    pol = np.polyfit(
        qubit_X_data.amplitude,
        np.unwrap(phase) / 2 / np.pi / data.flux_pulse_duration,
        deg=2,
    )
    print(pol)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=pol[2]
            + qubit_X_data.amplitude * pol[1]
            + qubit_X_data.amplitude**2 * pol[0],
            name="Fit Detuning [GHz]",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        xaxis3_title="Flux pulse amplitude [a.u.]",
        yaxis1_title="Prob of 1",
        yaxis2_title="Phase [rad]",
        yaxis3_title="Detuning [GHz]",
    )
    return [fig], fitting_report


cryoscope_amplitude = Routine(_acquisition, _fit, _plot)
