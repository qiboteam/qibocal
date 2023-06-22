from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine


@dataclass
class TimeOfFlightReadoutParameters(Parameters):
    """TimeOfFlightReadout runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TimeOfFlightReadoutResults(Results):
    """TimeOfFlightReadout outputs."""

    TimeOfFlightReadout: Dict[QubitId, float] = field(
        metadata=dict(update="time_of_flight")
    )
    """Time of flight"""


TimeOfFlightReadoutType = np.dtype([("samples", np.float64)])


@dataclass
class TimeOfFlightReadoutData(Data):
    """TimeOfFlightReadout acquisition outputs."""

    data: dict[QubitId, npt.NDArray[TimeOfFlightReadoutType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, samples):
        """Store output for single qubit."""
        shape = (1,) if np.isscalar(samples) else samples.shape
        ar = np.empty(shape, dtype=TimeOfFlightReadoutType)
        ar["samples"] = samples
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
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

    # execute the first pulse sequence
    results = platform.execute_pulse_sequence(
        sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    data = TimeOfFlightReadoutData()

    # retrieve and store the results for every qubit

    for qubit in qubits:
        samples = results[ro_pulses[qubit].serial].magnitude
        # store the results
        data.register_qubit(qubit, samples)

    # for qubit in qubits:
    #     r = results[ro_pulses[qubit].serial].serialize
    #     number_of_samples = len(r["MSR[V]"])
    #     r.update(
    #         {
    #             "qubit": [qubit] * number_of_samples,
    #             "sample": np.arange(number_of_samples),
    #         }
    #     )
    #     data.add_data_from_dict(r)

    # finally, save the remaining data
    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout."""
    return TimeOfFlightReadoutResults({})


def _plot(data: TimeOfFlightReadoutData, fit: TimeOfFlightReadoutResults, qubit):
    """Plotting function for TimeOfFlightReadout."""

    figures = []
    fitting_report = "No fitting data"
    fig = go.Figure()

    qubit_data = data[qubit]
    y = qubit_data.samples

    fig.add_trace(
        go.Scatter(
            # x=data.df["sample"],
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

    # fitting_report = fitting_report + (f"{qubit} | Time of flight : <br>")

    figures.append(fig)

    return figures, fitting_report


time_of_flight_readout = Routine(_acquisition, _fit, _plot)
"""TimeOfFlightReadout Routine object."""
