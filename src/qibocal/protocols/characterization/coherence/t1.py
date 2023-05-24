from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
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

    t1: Dict[Union[str, int], float] = field(metadata=dict(update="t1"))
    """T1 for each qubit."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]
    """Raw fitting output."""


class T1Data(DataUnits):
    """T1 acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"wait": "ns"},
            options=["qubit"],
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

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = T1Data()

    # repeat the experiment as many times as defined by software_averages
    # sweep the parameter
    for wait in ro_wait_range:
        for qubit in qubits:
            ro_pulses[qubit].start = qd_pulses[qubit].duration + wait

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(sequence)

        for ro_pulse in ro_pulses.values():
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[ro_pulse.serial].average.raw
            r.update(
                {
                    "wait[ns]": wait,
                    "qubit": ro_pulse.qubit,
                }
            )
            data.add(r)
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

    return figures, fitting_report


t1 = Routine(_acquisition, _fit, _plot)
"""T1 Routine object."""
