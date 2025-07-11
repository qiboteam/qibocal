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

__all__ = ["time_of_flight_readout"]

MINIMUM_TOF = 24
"""Minimum value for time of flight"""


@dataclass
class TimeOfFlightReadoutParameters(Parameters):
    """TimeOfFlightReadout runcard inputs."""

    readout_amplitude: Optional[int] = None
    """Amplitude of the readout pulse."""
    window_size: Optional[int] = 10
    """Window size for the moving average."""


@dataclass
class TimeOfFlightReadoutResults(Results):
    """TimeOfFlightReadout outputs."""

    time_of_flights: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


TimeOfFlightReadoutType = np.dtype([("samples", np.float64)])


@dataclass
class TimeOfFlightReadoutData(Data):
    """TimeOfFlightReadout acquisition outputs."""

    windows_size: int
    sampling_rate: int

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: TimeOfFlightReadoutParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TimeOfFlightReadoutData:
    """Data acquisition for time of flight experiment."""
    sequence = PulseSequence()
    ro_pulses = {}
    native = platform.natives.single_qubit
    ro_channels = []
    for qubit in targets:
        ro_channel, ro_pulse = native[qubit].MZ()[0]
        ro_channels.append(ro_channel)
        if params.readout_amplitude is not None:
            ro_pulse = replace(ro_pulse, amplitude=params.readout_amplitude)
        ro_pulses[qubit] = ro_pulse
        sequence.append((ro_channel, ro_pulse))
    results = platform.execute(
        [sequence],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
        updates=[{ro_channel: {"delay": MINIMUM_TOF} for ro_channel in ro_channels}],
    )

    data = TimeOfFlightReadoutData(
        windows_size=params.window_size, sampling_rate=platform.sampling_rate
    )

    # retrieve and store the results for every qubit
    for qubit in targets:
        samples = magnitude(results[ro_pulses[qubit].id])
        # store the results
        data.register_qubit(TimeOfFlightReadoutType, (qubit), dict(samples=samples))
    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout."""

    qubits = data.qubits
    time_of_flights = {}

    window_size = data.windows_size
    sampling_rate = data.sampling_rate

    for qubit in qubits:
        qubit_data = data[qubit]
        samples = qubit_data.samples
        window_size = int(len(qubit_data) / 10)
        th = (np.mean(samples[:window_size]) + np.mean(samples[:-window_size])) / 2
        delay = np.where(samples > th)[0][0]
        time_of_flight_readout = float(delay / sampling_rate + MINIMUM_TOF)
        time_of_flights[qubit] = time_of_flight_readout

    return TimeOfFlightReadoutResults(time_of_flights)


def _plot(
    data: TimeOfFlightReadoutData, target: QubitId, fit: TimeOfFlightReadoutResults
):
    """Plotting function for TimeOfFlightReadout."""

    figures = []
    fitting_report = ""
    fig = go.Figure()
    qubit_data = data[target]
    sampling_rate = data.sampling_rate
    y = qubit_data.samples

    fig.add_trace(
        go.Scatter(
            x=np.arange(0, len(y)) * sampling_rate + MINIMUM_TOF,
            y=y,
            textposition="bottom center",
            name="Expectation value",
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
            table_dict(target, "Time of flights [ns]", fit.time_of_flights[target])
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
