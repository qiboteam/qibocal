from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


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


class TimeOfFlightReadoutData(DataUnits):
    """TimeOfFlightReadout acquisition outputs."""

    def __init__(self):
        super().__init__(
            f"data",
            options=["qubit", "sample"],
        )


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
        r = results[ro_pulses[qubit].serial].serialize
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [qubit] * number_of_samples,
                "sample": np.arange(number_of_samples),
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data
    return data


def _fit(data: TimeOfFlightReadoutData) -> TimeOfFlightReadoutResults:
    """Post-processing function for TimeOfFlightReadout."""
    return TimeOfFlightReadoutResults({})


def _plot(data: TimeOfFlightReadoutData, fit: TimeOfFlightReadoutResults, qubit):
    """Plotting function for TimeOfFlightReadout."""
    figures = []
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    fitting_report = ""

    fig.add_trace(
        go.Scatter(
            x=data.df["sample"],
            y=data.df["MSR"].pint.to("uV").pint.magnitude,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    fitting_report = fitting_report + (f"{qubit} | Time of flight : <br>")

    figures.append(fig)

    return figures, fitting_report


time_of_flight_readout = Routine(_acquisition, _fit, _plot)
"""TimeOfFlightReadout Routine object."""
