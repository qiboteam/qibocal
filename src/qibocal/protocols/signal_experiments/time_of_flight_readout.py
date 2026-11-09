from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Pulse, PulseSequence, Rectangular

from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html
from qibocal.result import magnitude
from qibocal.update import replace

from .utils import _get_lo_frequency

__all__ = ["time_of_flight_readout"]

MINIMUM_TOF = 28
"""Minimum value for time of flight"""


def _window_size(intermediate_frequency: float, sampling_rate: float) -> int:
    """Number of samples spanning one period of the intermediate frequency.

    The baseline and signal levels are estimated by averaging over this window, so that
    any residual oscillation at the intermediate frequency (from imperfect I/Q balance)
    averages out.
    """
    period = 1e9 / abs(intermediate_frequency)  # [ns]
    return max(1, round(period * sampling_rate))


@dataclass
class TimeOfFlightReadoutParameters(Parameters):
    """TimeOfFlightReadout runcard inputs."""

    detuning: float = 20e6
    """Detuning with respect to corresponding LO frequency [Hz].

    Must be non-zero: the returned pulse is identified by its oscillation at this
    frequency, which also separates it from DC offsets and LO leakage.
    """
    duration: float = 1e3
    """Duration of the readout pulse and of the acquired trace [ns]."""
    readout_amplitude: float = 0.5
    """Amplitude of the readout pulse."""

    def __post_init__(self):
        if self.detuning == 0:
            raise ValueError("`detuning` must be non-zero.")


@dataclass
class TimeOfFlightReadoutResults(Results):
    """TimeOfFlightReadout outputs."""

    time_of_flights: dict[QubitId, float]
    """Time of flight of each qubit [ns]."""


@dataclass
class TimeOfFlightReadoutData(Data):
    """TimeOfFlightReadout acquisition outputs."""

    sampling_rate: float
    """ADC sampling rate [GSps]."""
    intermediate_frequency: float
    """Detuning of the probe with respect to the LO [Hz]."""
    amplitude: float
    """Amplitude of the readout pulse."""
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
        probe = Pulse(
            duration=params.duration,
            amplitude=params.readout_amplitude,
            envelope=Rectangular(),
        )
        # the RAW trace length is set by the acquisition, so it has to span the probe
        ro_pulse = replace(
            ro_pulse,
            probe=probe,
            acquisition=replace(ro_pulse.acquisition, duration=params.duration),
        )
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

    assert platform.sampling_rate is not None
    data = TimeOfFlightReadoutData(
        sampling_rate=platform.sampling_rate,
        intermediate_frequency=params.detuning,
        amplitude=params.readout_amplitude,
    )
    # retrieve and store the results for every qubit
    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        data.data[qubit] = results[acq_handle]

    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout.

    The magnitude of the raw trace is compared against a threshold halfway between the
    baseline level (mean of the first window, before the pulse arrives) and the signal
    level (mean of the last window, after it has arrived). The time of flight is the
    time of the first sample above the threshold.
    """

    qubits = data.qubits
    time_of_flights = {}

    sampling_rate = data.sampling_rate
    window = _window_size(data.intermediate_frequency, sampling_rate)
    for qubit in qubits:
        samples = magnitude(data.data[qubit])
        if len(samples) < 2 * window:
            raise ValueError(
                f"Trace of {len(samples)} samples is too short for a window of "
                f"{window} samples: increase `duration` or `detuning`."
            )
        th = (samples[:window].mean() + samples[-window:].mean()) / 2
        # first sample above the threshold marks the arrival of the signal
        delay = np.flatnonzero(samples > th)[0]
        time_of_flights[qubit] = float(delay / sampling_rate + MINIMUM_TOF)
    return TimeOfFlightReadoutResults(time_of_flights)


def _plot(
    data: TimeOfFlightReadoutData, target: QubitId, fit: TimeOfFlightReadoutResults
):
    """Plotting function for TimeOfFlightReadout."""

    figures = []
    fitting_report = ""
    fig = go.Figure()
    time = np.arange(len(data.data[target])) / data.sampling_rate + MINIMUM_TOF

    fig.add_trace(
        go.Scatter(
            x=time,
            y=data.data[target][:, 0],
            textposition="bottom center",
            name="I",
            showlegend=True,
            legendgroup="groupI",
            mode="markers",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=data.data[target][:, 1],
            textposition="bottom center",
            name="Q",
            showlegend=True,
            legendgroup="groupQ",
            mode="markers",
        ),
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
                    data.amplitude,
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


time_of_flight_readout = Protocol(_acquisition, _fit, _plot, _update)
"""TimeOfFlightReadout Protocol object."""
