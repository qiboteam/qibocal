from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.utils import S_TO_NS


@dataclass
class TimeOfFlightReadoutParameters(Parameters):
    """TimeOfFlightReadout runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    window_size: Optional[int] = 10
    """Window size for the moving average."""


@dataclass
class TimeOfFlightReadoutResults(Results):
    """TimeOfFlightReadout outputs."""

    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


TimeOfFlightReadoutType = np.dtype([("samples", np.float64)])


@dataclass
class TimeOfFlightReadoutData(Data):
    """TimeOfFlightReadout acquisition outputs."""

    windows_size: int
    sampling_rate: int

    data: dict[QubitId, npt.NDArray[TimeOfFlightReadoutType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, samples):
        """Store output for single qubit."""
        ar = np.empty(samples.shape, dtype=TimeOfFlightReadoutType)
        ar["samples"] = samples
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: TimeOfFlightReadoutParameters, platform: Platform, qubits: Qubits
) -> TimeOfFlightReadoutData:
    """Data acquisition for time of flight experiment."""

    sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        sequence.add(ro_pulses[qubit])

    results = platform.execute_pulse_sequence(
        sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    data = TimeOfFlightReadoutData(params.window_size, platform.sampling_rate)

    # retrieve and store the results for every qubit
    for qubit in qubits:
        samples = results[ro_pulses[qubit].serial].magnitude
        # store the results
        data.register_qubit(qubit, samples)

    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout."""

    qubits = data.qubits
    fitted_parameters = {}

    window_size = data.windows_size
    sampling_rate = data.sampling_rate

    for qubit in qubits:
        qubit_data = data[qubit]
        # Calculate moving average change per element
        moving_average_deltas = np.ediff1d(
            np.convolve(
                qubit_data.samples, np.ones(window_size) / window_size, mode="valid"
            )
        )

        max_average_change = np.argmax(moving_average_deltas)
        time_of_flight_readout = max_average_change / sampling_rate
        fitted_parameters[qubit] = time_of_flight_readout

    return TimeOfFlightReadoutResults(fitted_parameters)


def _plot(data: TimeOfFlightReadoutData, fit: TimeOfFlightReadoutResults, qubit):
    """Plotting function for TimeOfFlightReadout."""

    figures = []
    fitting_report = ""
    fig = go.Figure()

    qubit_data = data[qubit]
    sampling_rate = data.sampling_rate
    y = qubit_data.samples

    fig.add_trace(
        go.Scatter(
            y=y,
            textposition="bottom center",
            name="Expectation value",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    fig.add_vline(
        x=fit.fitted_parameters[qubit] * sampling_rate,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )

    fitting_report += (
        f"{qubit} | Time of flight(ns) : {fit.fitted_parameters[qubit] * S_TO_NS}<br>"
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


time_of_flight_readout = Routine(_acquisition, _fit, _plot)
"""TimeOfFlightReadout Routine object."""
