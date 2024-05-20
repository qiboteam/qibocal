from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Custom, PulseSequence, ReadoutPulse
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Qubits, Routine
from qibocal.protocols.characterization.coherence import t1_signal
from qibocal.protocols.characterization.coherence.utils import (
    exp_decay,
    exponential_fit,
)
from qibocal.protocols.characterization.utils import table_dict, table_html
from qibocal.protocols.characterization.z import t2


@dataclass
class T2SignalParameters(t2.T2Parameters):
    """T2Signal runcard inputs."""


@dataclass
class T2SignalResults(t2.T2Results):
    """T2Signal outputs."""


class T2SignalData(t1_signal.T1SignalData):
    """T2Signal acquisition outputs."""


def _acquisition(
    params: T2SignalParameters,
    platform: Platform,
    qubits: Qubits,
) -> T2SignalData:
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
        unpadded_ro_pulse = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        padded_ro_pulse = ReadoutPulse(
            start=unpadded_ro_pulse.start - RX90_pulses2[qubit].duration,
            duration=unpadded_ro_pulse.duration + RX90_pulses2[qubit].duration,
            amplitude=unpadded_ro_pulse.amplitude,
            frequency=unpadded_ro_pulse.frequency,
            relative_phase=unpadded_ro_pulse.relative_phase,
            shape=Custom(
                envelope_i=np.concatenate(
                    (
                        np.zeros(RX90_pulses2[qubit].duration),
                        unpadded_ro_pulse.envelope_waveform_i.data
                        / unpadded_ro_pulse.amplitude,
                    )
                ),
                envelope_q=np.concatenate(
                    (
                        np.zeros(RX90_pulses2[qubit].duration),
                        unpadded_ro_pulse.envelope_waveform_q.data
                        / unpadded_ro_pulse.amplitude,
                    )
                ),
            ),
            channel=unpadded_ro_pulse.channel,
            qubit=unpadded_ro_pulse.qubit,
        )
        ro_pulses[qubit] = padded_ro_pulse

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

    data = T2SignalData()

    sweeper = Sweeper(
        Parameter.start,
        waits,
        [RX90_pulses2[qubit] for qubit in qubits]
        + [ro_pulses[qubit] for qubit in qubits],
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
        data.register_qubit(
            t1_signal.CoherenceType,
            (qubit),
            dict(wait=waits, signal=result.magnitude, phase=result.phase),
        )
    return data


def _fit(data: T2SignalData) -> T2SignalResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 - p_1 e^{-x p_2}.
    """
    t2s, fitted_parameters = exponential_fit(data)
    return T2SignalResults(t2s, fitted_parameters)


def _plot(data: T2SignalData, qubit, fit: T2SignalResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = None

    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.wait,
            y=qubit_data.signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        )
    )

    if fit is not None:
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
                y=exp_decay(
                    waitrange,
                    *params,
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(qubit, "T2 [ns]", np.round(fit.t2[qubit]))
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: T2SignalResults, platform: Platform, qubit: QubitId):
    update.t2(results.t2[qubit], platform, qubit)


t2_signal = Routine(_acquisition, _fit, _plot, _update)
"""T2Signal Routine object."""