from dataclasses import dataclass
from typing import Union

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ...result import magnitude, phase
from ..ramsey.utils import ramsey_sequence
from ..utils import readout_frequency, table_dict, table_html
from . import utils
from .t1_signal import T1SignalData

__all__ = ["t2_signal", "update_t2", "T2SignalData", "T2SignalParameters"]


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


class T2SignalData(T1SignalData):
    """T2Signal acquisition outputs."""

    t2: dict[QubitId, float]
    """T2 for each qubit [ns]."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


def _acquisition(
    params: T2SignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> T2SignalData:
    """Data acquisition for T2 experiment.

    In this protocol the y axis is the magnitude of signal in the IQ plane.

    """

    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    sequence, delays = ramsey_sequence(platform, targets)

    data = T2SignalData()

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=waits,
        pulses=delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=(
            AveragingMode.SINGLESHOT if params.single_shot else AveragingMode.CYCLIC
        ),
    )

    for q in targets:
        ro_pulse = list(sequence.channel(platform.qubits[q].acquisition))[-1]
        result = results[ro_pulse.id]
        signal = magnitude(result)
        if params.single_shot:
            _waits = np.array(len(signal) * [waits])
        else:
            _waits = waits
        data.register_qubit(
            utils.CoherenceType,
            (q),
            dict(wait=_waits, signal=signal, phase=phase(result)),
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


def update_t2(results: T2SignalResults, platform: CalibrationPlatform, target: QubitId):
    update.t2(results.t2[target], platform, target)


t2_signal = Routine(_acquisition, _fit, _plot, update_t2)
"""T2Signal Routine object."""
