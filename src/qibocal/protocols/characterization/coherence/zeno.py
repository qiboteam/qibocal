from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..utils import V_TO_UV
from . import utils


@dataclass
class ZenoParameters(Parameters):
    """Zeno runcard inputs."""

    n_ros: int
    "Number of readout pulses"
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


ZenoType = np.dtype([("msr", np.float64), ("phase", np.float64)])
"""Custom dtype for Zeno."""


@dataclass
class ZenoData(Data):
    data: dict[tuple[QubitId, int], npt.NDArray[ZenoType]] = field(default_factory=dict)
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

    zeno_t1: dict[QubitId, float] = field(metadata=dict(update="t1"))
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


def _acquisition(
    params: ZenoParameters,
    platform: Platform,
    qubits: Qubits,
) -> ZenoData:
    """
    In a T1_Zeno experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.
    Args:
        platform (:class:`qibolab.platforms.abstract.Platform`): custom abstract platform on which we perform the calibration.
        qubits (dict): dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # create sequence of pulses:
    sequence = PulseSequence()
    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(RX_pulses[qubit])
        start = RX_pulses[qubit].finish
        ro_pulses[qubit] = []
        for _ in range(params.n_ros):
            ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
            start += ro_pulse.duration
            sequence.add(ro_pulse)
            ro_pulses[qubit].append(ro_pulse)

    # create a DataUnits object to store the results
    data = ZenoData()

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
            data.register_qubit(
                qubit=qubit, msr=result.magnitude, phase=result.phase  # , n_ros=i
            )

    return data


def _fit(data: ZenoData) -> ZenoResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    t1s, fitted_parameters = utils.exponential_fit_zeno(data)

    return ZenoResults(t1s, fitted_parameters)


def _plot(data: ZenoData, fit: ZenoResults, qubit):
    """Plotting function for T1 experiment."""

    figures = []
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data[qubit]
    n_ros = np.arange(1, len(qubit_data.msr) + 1)

    fig.add_trace(
        go.Scatter(
            x=n_ros,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    waitrange = np.linspace(
        min(n_ros),
        max(n_ros),
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
    fitting_report = fitting_report + (
        f"{qubit} | t1: {fit.zeno_t1[qubit]:,.0f} readout pulses.<br><br>"
    )
    # FIXME: Pulse duration (+ time of flight ?)
    fitting_report = fitting_report + (
        f"{qubit} | t1: {fit.zeno_t1[qubit]*2000:,.0f} ns.<br><br>"
    )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Number of readouts",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


zeno = Routine(_acquisition, _fit, _plot)
