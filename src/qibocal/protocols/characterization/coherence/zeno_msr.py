from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..utils import V_TO_UV, table_dict, table_html
from . import utils


@dataclass
class ZenoParameters(Parameters):
    """Zeno runcard inputs."""

    readouts: int
    "Number of readout pulses"
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


ZenoType = np.dtype([("msr", np.float64), ("phase", np.float64)])
"""Custom dtype for Zeno."""


@dataclass
class ZenoData(Data):
    readout_duration: dict[QubitId, float] = field(default_factory=dict)
    """Readout durations for each qubit"""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, msr, phase):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=ZenoType)
        ar["msr"] = msr
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


@dataclass
class ZenoResults(Results):
    """Zeno outputs."""

    zeno_t1: dict[QubitId, int]
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


def _acquisition(
    params: ZenoParameters,
    platform: Platform,
    qubits: Qubits,
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
    for qubit in qubits:
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
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        for ro_pulse in ro_pulses[qubit]:
            result = results[ro_pulse.serial]
            data.register_qubit(qubit=qubit, msr=result.magnitude, phase=result.phase)
    return data


def _fit(data: ZenoData) -> ZenoResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """

    t1s, fitted_parameters = utils.exponential_fit(data, zeno=True)

    return ZenoResults(t1s, fitted_parameters)


def _plot(data: ZenoData, fit: ZenoResults, qubit):
    """Plotting function for T1 experiment."""
    figures = []
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data[qubit]
    readouts = np.arange(1, len(qubit_data.msr) + 1)

    fig.add_trace(
        go.Scatter(
            x=readouts,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    if fit is not None:
        fitting_report = ""
        waitrange = np.linspace(
            min(readouts),
            max(readouts),
            2 * len(qubit_data),
        )
        params = fit.fitted_parameters[qubit]
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
                qubit,
                ["T1", "Readout Pulse"],
                [
                    np.round(fit.zeno_t1[qubit]),
                    np.round(fit.zeno_t1[qubit] * data.readout_duration[qubit]),
                ],
            )
        )
        # FIXME: Pulse duration (+ time of flight ?)

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Number of readouts",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ZenoResults, platform: Platform, qubit: QubitId):
    update.t1(results.zeno_t1[qubit], platform, qubit)


zeno_msr = Routine(_acquisition, _fit, _plot, _update)
