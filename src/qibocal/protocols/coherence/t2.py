from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Platform, Sweeper

from qibocal.auto.operation import QubitId, Routine

from ...result import probability
from ..ramsey.utils import ramsey_sequence
from ..utils import table_dict, table_html
from . import t1, t2_signal, utils


@dataclass
class T2Parameters(t2_signal.T2SignalParameters):
    """T2 runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""


@dataclass
class T2Results(t2_signal.T2SignalResults):
    """T2 outputs."""

    chi2: Optional[dict[QubitId, tuple[float, Optional[float]]]] = field(
        default_factory=dict
    )
    """Chi squared estimate mean value and error."""


class T2Data(t1.T1Data):
    """T2 acquisition outputs."""


def _acquisition(
    params: T2Parameters,
    platform: Platform,
    targets: list[QubitId],
) -> T2Data:
    """Data acquisition for T2 experiment."""

    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    sequence, delays = ramsey_sequence(platform, targets)

    data = T2Data()

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=waits,
        pulses=delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        probs = probability(results[ro_pulse.id], state=1)
        errors = np.sqrt(probs * (1 - probs) / params.nshots)
        data.register_qubit(
            t1.CoherenceProbType, (qubit), dict(wait=waits, prob=probs, error=errors)
        )
    return data


def _fit(data: T2Data) -> T2Results:
    """The used model is

    .. math::

        y = p_0 - p_1 e^{-x p_2}.
    """
    t2s, fitted_parameters, pcovs, chi2 = utils.exponential_fit_probability(data)
    return T2Results(t2s, fitted_parameters, pcovs, chi2)


def _plot(data: T2Data, target: QubitId, fit: T2Results = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fitting_report = ""
    qubit_data = data[target]
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

        params = fit.fitted_parameters[target]
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
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "T2 [ns]",
                    "chi2 reduced",
                ],
                [fit.t2[target], fit.chi2[target]],
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


t2 = Routine(_acquisition, _fit, _plot, t2_signal._update)
"""T2 Routine object."""
