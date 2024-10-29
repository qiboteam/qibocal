from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Platform, PulseSequence, Readout

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine

from ...result import magnitude, phase
from ..utils import table_dict, table_html
from . import utils


@dataclass
class ZenoSignalParameters(Parameters):
    """Zeno runcard inputs."""

    readouts: int
    "Number of readout pulses"


ZenoSignalType = np.dtype([("signal", np.float64), ("phase", np.float64)])
"""Custom dtype for Zeno."""


@dataclass
class ZenoSignalData(Data):

    readout_duration: dict[QubitId, float] = field(default_factory=dict)
    """Readout durations for each qubit"""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class ZenoSignalResults(Results):
    """Zeno outputs."""

    zeno_t1: dict[QubitId, int]
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""


def zeno_sequence(
    platform: Platform, targets: list[QubitId], readouts: int
) -> tuple[PulseSequence, dict[QubitId, int]]:
    """Generating sequence for Zeno experiment."""

    sequence = PulseSequence()
    readout_duration = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]
        _, ro_pulse = natives.MZ()[0]
        readout_duration[q] = ro_pulse.duration
        sequence |= natives.RX()

        for _ in range(readouts):
            sequence |= natives.MZ()

    return sequence, readout_duration


def _acquisition(
    params: ZenoSignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ZenoSignalData:
    """
    In a T1_Zeno experiment, we measure an excited qubit repeatedly. Due to decoherence processes,
    it is possible that, at the time of measurement, the qubit will not be excited anymore.
    The quantum zeno effect consists of measuring allowing a particle's time evolution to be slowed
    down by measuring it frequently enough. However, in the experiments we see that due the QND-ness of the readout
    pulse that the qubit decoheres faster.
    Reference: https://link.aps.org/accepted/10.1103/PhysRevLett.118.240401.
    """

    sequence, ro_pulse_duration = zeno_sequence(platform, targets, params.readouts)
    data = ZenoSignalData(readout_duration=ro_pulse_duration)

    results = platform.execute(
        [sequence],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in targets:
        res = []
        readouts = [
            pulse
            for pulse in sequence.channel(platform.qubits[qubit].acquisition)
            if isinstance(pulse, Readout)
        ]
        for i in range(params.readouts):
            ro_pulse = readouts[i]
            res.append(results[ro_pulse.id])

        data.register_qubit(
            utils.CoherenceType,
            (qubit),
            dict(
                wait=np.arange(params.readouts) + 1,
                signal=magnitude(res),
                phase=phase(res),
            ),
        )
    return data


def _fit(data: ZenoSignalData) -> ZenoSignalResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """

    t1s, fitted_parameters, pcovs = utils.exponential_fit(data, zeno=True)

    return ZenoSignalResults(t1s, fitted_parameters, pcovs)


def _plot(data: ZenoSignalData, fit: ZenoSignalResults, target: QubitId):
    """Plotting function for T1 experiment."""
    figures = []
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data[target]
    readouts = np.arange(1, len(qubit_data.signal) + 1)

    fig.add_trace(
        go.Scatter(
            x=readouts,
            y=qubit_data.signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        )
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
                ["T1", "Readout Pulse"],
                [
                    np.round(fit.zeno_t1[target][0]),
                    np.round(fit.zeno_t1[target][0] * data.readout_duration[target]),
                ],
            )
        )
        # FIXME: Pulse duration (+ time of flight ?)

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Number of readouts",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ZenoSignalResults, platform: Platform, qubit: QubitId):
    update.t1(results.zeno_t1[qubit], platform, qubit)


zeno_signal = Routine(_acquisition, _fit, _plot, _update)
