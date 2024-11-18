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

    fitted_parameters: dict[QubitId, dict[str, float]]
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
    for qubit in targets:
        ro_channel, ro_pulse = native[qubit].MZ()[0]
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
            y=y,
            textposition="bottom center",
            name="Expectation value",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Sample",
        yaxis_title="Signal [a.u.]",
    )
    if fit is not None:
        fig.add_vline(
            x=fit.fitted_parameters[target] * sampling_rate,
            line_width=2,
            line_dash="dash",
            line_color="grey",
        )
        fitting_report = table_html(
            table_dict(target, "Time of flights [ns]", fit.fitted_parameters[target])
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Sample",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


time_of_flight_readout = Routine(_acquisition, _fit, _plot)
"""TimeOfFlightReadout Routine object."""
