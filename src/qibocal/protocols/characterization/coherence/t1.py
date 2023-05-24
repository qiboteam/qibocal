from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color

from . import utils


@dataclass
class T1Parameters(Parameters):
    """T1 runcard inputs."""

    delay_before_readout_start: int
    """Initial delay before readout (ns)."""
    delay_before_readout_end: int
    """Final delay before readout (ns)."""
    delay_before_readout_step: int
    """Step delay before readout (ns)."""


@dataclass
class T1Results(Results):
    """T1 outputs."""

    t1: Dict[List[Tuple], str] = field(metadata=dict(update="t1"))
    """T1 for each qubit."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitting output."""


class T1Data(DataUnits):
    """T1 acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"wait": "ns"},
            options=["qubit", "probability"],
        )


def _acquisition(
    params: T1Parameters, platform: AbstractPlatform, qubits: Qubits
) -> T1Data:
    r"""Data acquisition for T1 experiment.
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        params:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points
    """

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    sweeper = Sweeper(
        Parameter.delay,
        ro_wait_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = T1Data()

    # sweep the parameter
    # execute the pulse sequence
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in qubits:
        r = results[ro_pulses[qubit].serial].serialize
        r.update(
            {
                "wait[ns]": ro_wait_range,
                "qubit": len(ro_wait_range) * [qubit],
                "probability": np.abs(
                    results[ro_pulses[qubit].serial].voltage_i
                    + 1j * results[ro_pulses[qubit].serial].voltage_q
                    - complex(platform.qubits[qubit].mean_gnd_states)
                )
                / np.abs(
                    complex(platform.qubits[qubit].mean_exc_states)
                    - complex(platform.qubits[qubit].mean_gnd_states)
                ),
            }
        )
        data.add_data_from_dict(r)
    return data


def _fit(data: T1Data) -> T1Results:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    t1s, fitted_parameters = utils.exponential_fit(data)

    return T1Results(t1s, fitted_parameters)


def _plot(data: T1Data, fit: T1Results, qubit):
    """Plotting function for T1 experiment."""

    figures = []
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data.df[data.df["qubit"] == qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.to("ns").pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    #  add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(qubit_data["wait"].pint.to("ns").pint.magnitude),
            max(qubit_data["wait"].pint.to("ns").pint.magnitude),
            2 * len(qubit_data),
        )

        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=utils.exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(2),
            )
        )
        fitting_report = fitting_report + (
            f"{qubit} | t1: {fit.t1[qubit]:,.0f} ns.<br><br>"
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    # Plot the probability
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.to("ns").pint.magnitude,
            y=qubit_data["probability"],
            marker_color=get_color(0),
            opacity=1,
            name="Probability",
            showlegend=True,
            legendgroup="Probability",
        )
    )
    fig2.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="Probability",
    )
    figures.append(fig2)

    return figures, fitting_report


t1 = Routine(_acquisition, _fit, _plot)
"""T1 Routine object."""
