from dataclasses import dataclass
from typing import Union

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Delay
from qibolab.sequence import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Parameters, QubitId, Results, Routine

from ..utils import table_dict, table_html
from . import t1_signal, t2, utils


@dataclass
class T2SignalParameters(Parameters):
    """T2Signal runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    single_shot: bool = False
    """If ``True`` save single shot signal data."""


@dataclass
class T2SignalResults(Results):
    """T2Signal outputs."""

    t2: dict[QubitId, Union[float, list[float]]]
    """T2 for each qubit [ns]."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""


class T2SignalData(t1_signal.T1SignalData):
    """T2Signal acquisition outputs."""

    t2: dict[QubitId, float]
    """T2 for each qubit [ns]."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


def _acquisition(
    params: T2SignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> T2SignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    qd_delays = {}
    ro_delays = {}
    sequence = PulseSequence()
    for q in targets:
        qubit = platform.qubits[q]
        qd_sequence = qubit.native_gates.RX.create_sequence(theta=np.pi / 2, phi=0)
        ro_sequence = qubit.native_gates.MZ.create_sequence()
        qd_delays[q] = Delay(duration=16)
        ro_delays[q] = Delay(duration=16)
        qd_pulse = qd_sequence[qubit.drive.name][0]
        ro_pulses[q] = ro_sequence[qubit.measure.name][0]
        sequence.extend(qd_sequence)
        sequence[qubit.drive.name].append(qd_delays[q])
        sequence.extend(qd_sequence)
        sequence[qubit.measure.name].append(Delay(duration=2 * qd_pulse.duration))
        sequence[qubit.measure.name].append(ro_delays[q])
        sequence.extend(ro_sequence)

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    sweeper = Sweeper(
        Parameter.duration,
        waits,
        [qd_delays[q] for q in targets] + [ro_delays[q] for q in targets],
        type=SweeperType.ABSOLUTE,
    )

    # execute the sweep
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=(
                AveragingMode.SINGLESHOT if params.single_shot else AveragingMode.CYCLIC
            ),
        ),
        sweeper,
    )

    data = T2SignalData()
    for q in targets:
        result = results[ro_pulses[q].id]
        if params.single_shot:
            _waits = np.array(len(result.magnitude) * [waits])
        else:
            _waits = waits
        data.register_qubit(
            utils.CoherenceType,
            (q),
            dict(wait=_waits, signal=result.magnitude, phase=result.phase),
        )
    return data


def _fit(data: T2SignalData) -> T2SignalResults:
    """The used model is

    .. math::

        y = p_0 - p_1 e^{-x p_2}.
    """
    data = data.average

    t2s, fitted_parameters, pcovs = utils.exponential_fit(data)
    return T2SignalResults(t2s, fitted_parameters, pcovs)


def _plot(data: T2SignalData, target: QubitId, fit: T2SignalResults = None):
    """Plotting function for Ramsey Experiment."""
    data = data.average

    figures = []
    fig = go.Figure()
    fitting_report = None

    qubit_data = data[target]

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
                target, ["T2 [ns]"], [np.round(fit.t2[target])], display_error=True
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: T2SignalResults, platform: Platform, target: QubitId):
    update.t2(results.t2[target], platform, target)


t2_signal = Routine(_acquisition, _fit, _plot, _update)
"""T2Signal Routine object."""
