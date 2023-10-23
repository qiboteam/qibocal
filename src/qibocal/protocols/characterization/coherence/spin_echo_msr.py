from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Qubits, Routine

from ..utils import V_TO_UV, table_dict, table_html
from . import spin_echo
from .t1_msr import CoherenceType, T1MSRData
from .utils import exp_decay, exponential_fit


@dataclass
class SpinEchoMSRParameters(spin_echo.SpinEchoParameters):
    """SpinEcho MSR runcard inputs."""


@dataclass
class SpinEchoMSRResults(spin_echo.SpinEchoResults):
    """SpinEchoMSR outputs."""


class SpinEchoMSRData(T1MSRData):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoMSRParameters,
    platform: Platform,
    qubits: Qubits,
) -> SpinEchoMSRData:
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

    data = SpinEchoMSRData()

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

        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                CoherenceType,
                (qubit),
                dict(
                    wait=np.array([wait]),
                    msr=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )
    return data


def _fit(data: SpinEchoMSRData) -> SpinEchoMSRResults:
    """Post-processing for SpinEcho."""
    t2Echos, fitted_parameters = exponential_fit(data)

    return SpinEchoMSRResults(t2Echos, fitted_parameters)


def _plot(data: SpinEchoMSRData, qubit, fit: SpinEchoMSRResults = None):
    """Plotting for SpinEcho"""

    figures = []
    fig = go.Figure()

    # iterate over multiple data folders
    fitting_report = None

    qubit_data = data[qubit]
    waits = qubit_data.wait

    fig.add_trace(
        go.Scatter(
            x=waits,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
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
            table_dict(qubit, "T2 Spin Echo [ns]", np.round(fit.t2_spin_echo[qubit]))
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: SpinEchoMSRResults, platform: Platform, qubit: QubitId):
    update.t2_spin_echo(results.t2_spin_echo[qubit], platform, qubit)


spin_echo_msr = Routine(_acquisition, _fit, _plot, _update)
"""SpinEcho Routine object."""
