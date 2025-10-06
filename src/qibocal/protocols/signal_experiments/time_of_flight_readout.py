from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html
from qibocal.result import magnitude
from qibocal.update import replace

from .utils import _get_lo_frequency

__all__ = ["time_of_flight_readout"]

MINIMUM_TOF = 24
"""Minimum value for time of flight"""


@dataclass
class TimeOfFlightReadoutParameters(Parameters):
    """TimeOfFlightReadout runcard inputs."""

    detuning: float = 10e6
    """Detuning with respect to corresponding LO frequency [Hz]."""
    readout_amplitude: Optional[int] = None
    """Amplitude of the readout pulse."""
    window_size: Optional[int] = 10
    """Window size for the moving average."""


@dataclass
class TimeOfFlightReadoutResults(Results):
    """TimeOfFlightReadout outputs."""

    time_of_flights: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


@dataclass
class TimeOfFlightReadoutData(Data):
    """TimeOfFlightReadout acquisition outputs."""

    windows_size: int
    sampling_rate: int
    intermediate_frequency: float
    amplitude: dict[QubitId, float] = field(default_factory=dict)
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: TimeOfFlightReadoutParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TimeOfFlightReadoutData:
    """Data acquisition for time of flight experiment."""
    sequence = PulseSequence()
    native = platform.natives.single_qubit
    for qubit in targets:
        ro_channel, ro_pulse = native[qubit].MZ()[0]
        if params.readout_amplitude is not None:
            probe = replace(ro_pulse.probe, amplitude=params.readout_amplitude)
            ro_pulse = replace(ro_pulse, probe=probe)
        sequence.append((ro_channel, ro_pulse))

    results = platform.execute(
        [sequence],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
        updates=[
            {
                platform.qubits[qubit].acquisition: {"delay": MINIMUM_TOF},
                platform.qubits[qubit].probe: {
                    "frequency": _get_lo_frequency(platform, qubit) + params.detuning,
                },
            }
            for qubit in targets
        ],
    )

    data = TimeOfFlightReadoutData(
        windows_size=params.window_size,
        sampling_rate=platform.sampling_rate,
        intermediate_frequency=params.detuning,
        amplitude={
            qubit: list(sequence.channel(platform.qubits[qubit].acquisition))[
                -1
            ].probe.amplitude
            for qubit in targets
        },
    )
    # retrieve and store the results for every qubit
    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        data.data[qubit] = results[acq_handle]

    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout."""

    qubits = data.qubits
    time_of_flights = {}

    window_size = data.windows_size
    sampling_rate = data.sampling_rate
    for qubit in qubits:
        delays = []
        for i in range(2):
            samples = data.data[qubit][:, i]
            window_size = int(len(samples) / 10)

            for feat in ["min", "max"]:
                th = (
                    getattr(np, feat)(samples[:window_size])
                    + getattr(np, feat)(samples[:-window_size])
                ) / 2
                # try-expect in order to handle sporadic failing with mock
                try:
                    delay = np.where(samples < th if feat == "min" else samples > th)[
                        0
                    ][0]
                except IndexError:
                    delay = 0
                delays.append(delay)
        time_of_flight_readout = float(min(delays) / sampling_rate + MINIMUM_TOF)
        time_of_flights[qubit] = time_of_flight_readout
    return TimeOfFlightReadoutResults(time_of_flights)


def _plot(
    data: TimeOfFlightReadoutData, target: QubitId, fit: TimeOfFlightReadoutResults
):
    """Plotting function for TimeOfFlightReadout."""

    figures = []
    fitting_report = ""
    fig = go.Figure()
    sampling_rate = data.sampling_rate
    y = magnitude(data.data[target])

    fig.add_trace(
        go.Scatter(
            x=np.arange(0, len(y)) * sampling_rate + MINIMUM_TOF,
            y=data.data[target][:, 0],
            textposition="bottom center",
            name="I",
            showlegend=True,
            legendgroup="group1",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, len(y)) * sampling_rate + MINIMUM_TOF,
            y=data.data[target][:, 1],
            textposition="bottom center",
            name="Q",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )
    if fit is not None:
        fig.add_vline(
            x=fit.time_of_flights[target],
            line_width=2,
            line_dash="dash",
            line_color="grey",
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Intermediate Frequency [Hz]",
                    "Readout amplitude [a.u.]",
                    "Time of flights [ns]",
                ],
                [
                    data.intermediate_frequency,
                    data.amplitude[target],
                    fit.time_of_flights[target],
                ],
            )
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: TimeOfFlightReadoutResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    ro_channel = platform.qubits[qubit].acquisition
    platform.update({f"configs.{ro_channel}.delay": results.time_of_flights[qubit]})


time_of_flight_readout = Routine(_acquisition, _fit, _plot, _update)
"""TimeOfFlightReadout Routine object."""
