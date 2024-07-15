from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Custom, PulseSequence, ReadoutPulse
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.coherence import t1
from qibocal.protocols.characterization.coherence.utils import (
    exp_decay,
    exponential_fit_probability,
)
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class T2Parameters(Parameters):
    """T2 runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""


@dataclass
class T2Results(Results):
    """T2 outputs."""

    t2: dict[QubitId, float]
    """T2 for each qubit [ns]."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    chi2: Optional[dict[QubitId, tuple[float, Optional[float]]]] = field(
        default_factory=dict
    )
    """Chi squared estimate mean value and error."""


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

    data = T2Data()

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
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    for qubit in qubits:
        probs = results[ro_pulses[qubit].serial].probability(state=1)
        errors = np.sqrt(probs * (1 - probs) / params.nshots)
        data.register_qubit(
            t1.CoherenceProbType, (qubit), dict(wait=waits, prob=probs, error=errors)
        )
    return data


def _fit(data: T2Data) -> T2Results:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 - p_1 e^{-x p_2}.
    """
    t2s, fitted_parameters, chi2 = exponential_fit_probability(data)
    return T2Results(t2s, fitted_parameters, chi2)


def _plot(data: T2Data, qubit, fit: T2Results = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
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
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
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
            table_dict(
                qubit,
                [
                    "T2 [ns]",
                    "chi2 reduced",
                ],
                [fit.t2[qubit], fit.chi2[qubit]],
                display_error=True,
            )
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: T2Results, platform: Platform, qubit: QubitId):
    update.t2(results.t2[qubit], platform, qubit)


t2 = Routine(_acquisition, _fit, _plot, _update)
"""T2 Routine object."""