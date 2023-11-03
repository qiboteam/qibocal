from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from ..utils import table_dict, table_html
from . import t1
from .utils import exp_decay, exponential_fit_probability


@dataclass
class SpinEchoParameters(Parameters):
    """SpinEcho runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between pulses [ns]."""
    delay_between_pulses_end: int
    """Final delay between pulses [ns]."""
    delay_between_pulses_step: int
    """Step delay between pulses (ns)."""


@dataclass
class SpinEchoResults(Results):
    """SpinEcho outputs."""

    t2_spin_echo: dict[QubitId, float]
    """T2 echo for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


class SpinEchoData(t1.T1Data):
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
    probs = {qubit: [] for qubit in qubits}
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
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
        )

        for qubit in qubits:
            prob = results[ro_pulses[qubit].serial].probability(state=0)
            probs[qubit].append(prob)

    for qubit in qubits:
        errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs[qubit]]
        data.register_qubit(
            t1.CoherenceProbType,
            (qubit),
            dict(wait=ro_wait_range, prob=probs[qubit], error=errors),
        )

    return data


def _fit(data: SpinEchoData) -> SpinEchoResults:
    """Post-processing for SpinEcho."""
    t2Echos, fitted_parameters = exponential_fit_probability(data)

    return SpinEchoResults(t2Echos, fitted_parameters)


def _plot(data: SpinEchoData, qubit, fit: SpinEchoResults = None):
    """Plotting for SpinEcho"""

    figures = []
    # iterate over multiple data folders
    fitting_report = ""

    qubit_data = data[qubit]
    waits = qubit_data.wait
    probs = qubit_data.prob
    error_bars = qubit_data.error

    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of 0",
                showlegend=True,
                legendgroup="Probability of 0",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=t1.COLORBAND,
                line=dict(color=t1.COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        # add fitting trace
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(qubit_data),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                ["T2 Spin Echo [ns]"],
                [fit.t2_spin_echo[qubit]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time (ns)",
        yaxis_title="Probability of State 0",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: SpinEchoResults, platform: Platform, qubit: QubitId):
    update.t2_spin_echo(results.t2_spin_echo[qubit], platform, qubit)


spin_echo = Routine(_acquisition, _fit, _plot, _update)
"""SpinEcho Routine object."""
