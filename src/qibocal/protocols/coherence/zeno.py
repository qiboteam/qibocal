from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Routine

from ..utils import table_dict, table_html
from . import t1, utils
from .zeno_signal import ZenoSignalParameters, ZenoSignalResults, _update


@dataclass
class ZenoParameters(ZenoSignalParameters):
    """Zeno runcard inputs."""


@dataclass
class ZenoData(t1.T1Data):
    readout_duration: dict[QubitId, float] = field(default_factory=dict)
    """Readout durations for each qubit"""


@dataclass
class ZenoResults(ZenoSignalResults):
    """Zeno outputs."""

    chi2: dict[QubitId, tuple[float, Optional[float]]]
    """Chi squared estimate mean value and error."""


def _acquisition(
    params: ZenoParameters,
    platform: Platform,
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

    # create sequence of pulses:
    sequence = PulseSequence()
    RX_pulses = {}
    ro_pulses = {}
    ro_pulse_duration = {}
    for qubit in targets:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(RX_pulses[qubit])
        start = RX_pulses[qubit].finish
        ro_pulses[qubit] = []
        for _ in range(params.readouts):
            ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
            start += ro_pulse.duration
            sequence.add(ro_pulse)
            ro_pulses[qubit].append(ro_pulse)
        ro_pulse_duration[qubit] = ro_pulse.duration

    # create a DataUnits object to store the results
    data = ZenoData(readout_duration=ro_pulse_duration)

    # execute the first pulse sequence
    results = platform.execute_pulse_sequence(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
    )

    # retrieve and store the results for every qubit
    probs = {qubit: [] for qubit in targets}
    for qubit in targets:
        for ro_pulse in ro_pulses[qubit]:
            probs[qubit].append(results[ro_pulse.serial].probability(state=1))
        errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs[qubit]]
        data.register_qubit(
            t1.CoherenceProbType,
            (qubit),
            dict(
                wait=np.arange(1, len(probs[qubit]) + 1),
                prob=probs[qubit],
                error=errors,
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
    readouts = np.arange(1, len(qubit_data.prob) + 1)

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
                fillcolor=t1.COLORBAND,
                line=dict(color=t1.COLORBAND_LINE),
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
                    fit.zeno_t1[target],
                    np.array(fit.zeno_t1[target]) * data.readout_duration[target],
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


zeno = Routine(_acquisition, _fit, _plot, _update)
