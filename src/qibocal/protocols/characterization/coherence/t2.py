from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from ..utils import V_TO_UV
from . import t1, utils


@dataclass
class T2Parameters(Parameters):
    """T2 runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class T2Results(Results):
    """T2 outputs."""

    t2: dict[QubitId, float] = field(metadata=dict(update="t2"))
    """T2 for each qubit (ns)."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


class T2Data(t1.T1Data):
    """T2 acquisition outputs."""


def _acquisition(
    params: T2Parameters,
    platform: Platform,
    qubits: Qubits,
) -> T2Data:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit,
            start=RX90_pulses1[qubit].finish,
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = T2Data()

    sweeper = Sweeper(
        Parameter.start,
        waits,
        [RX90_pulses2[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # execute the sweep
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(qubit, wait=waits, msr=result.magnitude, phase=result.phase)
    return data


def _fit(data: T2Data) -> T2Results:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 - p_1 e^{-x p_2}.
    """
    t2s, fitted_parameters = utils.exponential_fit(data)
    return T2Results(t2s, fitted_parameters)


def _plot(data: T2Data, fit: T2Results, qubit):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.wait,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    # add fitting trace
    waitrange = np.linspace(
        min(qubit_data.wait),
        max(qubit_data.wait),
        2 * len(qubit_data),
    )

    params = fit.fitted_parameters[qubit]
    fig.add_trace(
        go.Scatter(
            x=waitrange,
            y=utils.exp_decay(
                waitrange,
                *params,
            ),
            name="Fit",
            line=go.scatter.Line(dash="dot"),
        )
    )
    fitting_report = fitting_report + (
        f"{qubit} | T2: {fit.t2[qubit]:,.0f} ns.<br><br>"
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


t2 = Routine(_acquisition, _fit, _plot)
"""T2 Routine object."""
