from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import Data, QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html
from . import utils
from .t1_signal import T1SignalParameters, T1SignalResults, t1_sequence, update_t1

__all__ = ["CoherenceProbType", "T1Data", "t1"]


@dataclass
class T1Parameters(T1SignalParameters):
    """T1 runcard inputs."""


@dataclass
class T1Results(T1SignalResults):
    """T1 outputs."""

    chi2: Optional[dict[QubitId, list[float]]] = field(default_factory=dict)
    """Chi squared estimate mean value and error."""


CoherenceProbType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class T1Data(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: T1Parameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> T1Data:
    """Data acquisition for T1 experiment."""

    sequence, ro_pulses, pulses = t1_sequence(
        platform=platform,
        targets=targets,
    )

    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=ro_wait_range,
        pulses=pulses,
    )

    data = T1Data()

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        probs = probability(results[ro_pulses[qubit].id], state=1)
        errors = np.sqrt(probs * (1 - probs) / params.nshots)
        data.register_qubit(
            CoherenceProbType,
            (qubit),
            dict(wait=ro_wait_range, prob=probs, error=errors),
        )

    return data


def _fit(data: T1Data) -> T1Results:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    t1s, fitted_parameters, pcovs, chi2 = utils.exponential_fit_probability(data)
    return T1Results(t1s, fitted_parameters, pcovs, chi2)


def _plot(data: T1Data, target: QubitId, fit: T1Results = None):
    """Plotting function for T1 experiment."""

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
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(qubit_data),
        )

        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=utils.exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "T1 [ns]",
                    "chi2 reduced",
                ],
                [fit.t1[target], fit.chi2[target]],
                display_error=True,
            )
        )
    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report


t1 = Routine(_acquisition, _fit, _plot, update_t1)
"""T1 Routine object."""
