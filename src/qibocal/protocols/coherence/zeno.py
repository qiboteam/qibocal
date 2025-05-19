from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence, Readout

from qibocal import update
from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ...result import probability
from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html
from . import utils
from .t1 import CoherenceProbType, T1Data


@dataclass
class ZenoParameters(Parameters):
    """Zeno runcard inputs."""

    readouts: int
    "Number of readout pulses"


@dataclass
class ZenoResults(Results):
    """Zeno outputs."""

    zeno_t1: dict[QubitId, int]
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""
    chi2: dict[QubitId, tuple[float, Optional[float]]]
    """Chi squared estimate mean value and error."""


def zeno_sequence(
    platform: CalibrationPlatform, targets: list[QubitId], readouts: int
) -> tuple[PulseSequence, dict[QubitId, int]]:
    """Generating sequence for Zeno experiment."""

    sequence = PulseSequence()
    readout_duration = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]
        _, ro_pulse = natives.MZ()[0]
        readout_duration[q] = ro_pulse.duration
        qubit_sequence = natives.RX() | natives.MZ()
        for _ in range(readouts - 1):
            qubit_sequence += natives.MZ()
        sequence += qubit_sequence

    return sequence, readout_duration


@dataclass
class ZenoData(T1Data):
    readout_duration: dict[QubitId, float] = field(default_factory=dict)
    """Readout durations for each qubit"""


def _acquisition(
    params: ZenoParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ZenoData:
    """
    In a T1_Zeno experiment, we measure an excited qubit repeatedly. Due to decoherence processes,
    it is possible that, at the time of measurement, the qubit will not be excited anymore.
    The quantum zeno effect consists of measuring allowing a particle's time evolution to be slowed
    down by measuring it frequently enough. However, in the experiments we see that due the QND-ness of the readout
    pulse that the qubit decoheres faster.
    Reference: https://link.aps.org/accepted/10.1103/PhysRevLett.118.240401.
    """

    sequence, readout_duration = zeno_sequence(
        platform, targets, readouts=params.readouts
    )
    data = ZenoData(readout_duration=readout_duration)

    results = platform.execute(
        [sequence],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        probs = []
        readouts = [
            pulse
            for pulse in sequence.channel(platform.qubits[qubit].acquisition)
            if isinstance(pulse, Readout)
        ]
        for i in range(params.readouts):
            ro_pulse = readouts[i]
            probs.append(probability(results[ro_pulse.id], state=1))

        data.register_qubit(
            CoherenceProbType,
            (qubit),
            dict(
                wait=np.arange(params.readouts) + 1,
                prob=np.array(probs),
                error=np.array(
                    [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]
                ),
            ),
        )
    return data


def _fit(data: ZenoData) -> ZenoResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    t1s, fitted_parameters, pcovs, chi2 = utils.exponential_fit_probability(data)
    return ZenoResults(t1s, fitted_parameters, pcovs, chi2)


def _plot(data: ZenoData, fit: ZenoResults, target: QubitId):
    """Plotting function for T1 experiment."""

    figures = []
    fitting_report = ""
    qubit_data = data[target]
    probs = qubit_data.prob
    error_bars = qubit_data.error
    readouts = qubit_data.wait

    fig = go.Figure(
        [
            go.Scatter(
                x=readouts,
                y=probs,
                opacity=1,
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((readouts, readouts[::-1])),
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
        fitting_report = ""
        waitrange = np.linspace(
            min(readouts),
            max(readouts),
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
                ["T1 [ns]", "Readout Pulse [ns]", "chi2 reduced"],
                [
                    np.array(fit.zeno_t1[target]) * data.readout_duration[target],
                    (data.readout_duration[target], 0),
                    fit.chi2[target],
                ],
                display_error=True,
            )
        )
        # FIXME: Pulse duration (+ time of flight ?)

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Number of readouts",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ZenoResults, platform: CalibrationPlatform, qubit: QubitId):
    update.t1(results.zeno_t1[qubit], platform, qubit)


zeno = Routine(_acquisition, _fit, _plot, _update)
