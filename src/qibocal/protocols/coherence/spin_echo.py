from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from ..utils import table_dict, table_html
from . import t1
from .spin_echo_signal import SpinEchoSignalParameters, SpinEchoSignalResults, _update
from .utils import exp_decay, exponential_fit_probability, spin_echo_sequence


@dataclass
class SpinEchoParameters(SpinEchoSignalParameters):
    """SpinEcho runcard inputs."""


@dataclass
class SpinEchoResults(SpinEchoSignalResults):
    """SpinEcho outputs."""

    chi2: Optional[dict[QubitId, tuple[float, Optional[float]]]] = field(
        default_factory=dict
    )
    """Chi squared estimate mean value and error."""


class SpinEchoData(t1.T1Data):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SpinEchoData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    sequence, delays = spin_echo_sequence(platform, targets)

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=ro_wait_range,
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

    data = SpinEchoData()
    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        result = results[ro_pulse.id]
        prob = probability(result, state=0)
        error = np.sqrt(prob * (1 - prob) / params.nshots)
        data.register_qubit(
            t1.CoherenceProbType,
            (qubit),
            dict(
                wait=ro_wait_range,
                prob=prob,
                error=error,
            ),
        )

    return data


def _fit(data: SpinEchoData) -> SpinEchoResults:
    """Post-processing for SpinEcho."""
    t2Echos, fitted_parameters, pcovs, chi2 = exponential_fit_probability(data)
    return SpinEchoResults(t2Echos, fitted_parameters, pcovs, chi2)


def _plot(data: SpinEchoData, target: QubitId, fit: SpinEchoResults = None):
    """Plotting for SpinEcho"""

    figures = []
    # iterate over multiple data folders
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
        params = fit.fitted_parameters[target]

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
                target,
                ["T2 Spin Echo [ns]", "chi2 reduced"],
                [fit.t2_spin_echo[target], fit.chi2[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Probability of State 0",
    )

    figures.append(fig)

    return figures, fitting_report


spin_echo = Routine(_acquisition, _fit, _plot, _update)
"""SpinEcho Routine object."""
