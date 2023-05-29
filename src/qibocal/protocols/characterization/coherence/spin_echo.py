from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.plots.utils import get_color

from .t1 import T1Data
from .utils import exp_decay, exponential_fit


@dataclass
class SpinEchoParameters(Parameters):
    """SpinEcho runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between pulses [ns]."""
    delay_between_pulses_end: int
    """Final delay between pulses [ns]."""
    delay_between_pulses_step: int
    """Step delay between pulses (ns)."""
    qubits: Optional[list] = field(default_factory=list)
    """Local qubits (optional)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class SpinEchoResults(Results):
    """SpinEcho outputs."""

    t2_spin_echo: Dict[Union[str, int], float] = field(
        metadata=dict(update="t2_spin_echo")
    )
    """T2 echo for each qubit."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]
    """Raw fitting output."""


class SpinEchoData(T1Data):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoParameters,
    platform: Platform,
    qubits: Qubits,
) -> SpinEchoData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    ro_pulses = {}
    RX90_pulses1 = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX_pulses[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = SpinEchoData()

    # sweep the parameter
    for wait in ro_wait_range:
        # save data as often as defined by points

        for qubit in qubits:
            RX_pulses[qubit].start = RX90_pulses1[qubit].finish + wait
            RX90_pulses2[qubit].start = RX_pulses[qubit].finish + wait
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        for ro_pulse in ro_pulses.values():
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[ro_pulse.serial].serialize
            r.update(
                {
                    "wait[ns]": 2 * wait,
                    "qubit": ro_pulse.qubit,
                }
            )
            data.add_data_from_dict(r)
    return data


def _fit(data: SpinEchoData) -> SpinEchoResults:
    """Post-processing for SpinEcho."""
    t2Echos, fitted_parameters = exponential_fit(data)

    return SpinEchoResults(t2Echos, fitted_parameters)


def _plot(data: SpinEchoData, fit: SpinEchoResults, qubit: int):
    """Plotting for SpinEcho"""

    figures = []
    fig = go.Figure()

    # iterate over multiple data folders
    fitting_report = ""

    qubit_data = data.df[data.df["qubit"] == qubit]
    waits = data.df["wait"].pint.to("ns").pint.magnitude

    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.to("ns").pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
    )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(data),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            ),
        )

        fitting_report = fitting_report + (
            f"{qubit} | T2 Spin Echo: {fit.t2_spin_echo[qubit]:,.0f} ns.<br><br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


spin_echo = Routine(_acquisition, _fit, _plot)
"""SpinEcho Routine object."""
