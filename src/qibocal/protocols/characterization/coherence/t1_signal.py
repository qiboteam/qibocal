from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Routine

from ..utils import table_dict, table_html
from . import t1, utils


@dataclass
class T1SignalParameters(t1.T1Parameters):
    """T1 Signal runcard inputs."""


@dataclass
class T1SignalResults(t1.T1Results):
    """T1 Signal outputs."""


CoherenceType = np.dtype(
    [("wait", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class T1SignalData(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: T1SignalParameters, platform: Platform, targets: list[QubitId]
) -> T1SignalData:
    r"""Data acquisition for T1 experiment.
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        params:
        platform (Platform): Qibolab platform object
        targets (list): list of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points
    """

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in targets:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    sweeper = Sweeper(
        Parameter.start,
        ro_wait_range,
        [ro_pulses[qubit] for qubit in targets],
        type=SweeperType.ABSOLUTE,
    )

    data = T1SignalData()

    # sweep the parameter
    # execute the pulse sequence
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            CoherenceType,
            (qubit),
            dict(wait=ro_wait_range, signal=result.magnitude, phase=result.phase),
        )

    return data


def _fit(data: T1SignalData) -> T1SignalResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    t1s, fitted_parameters = utils.exponential_fit(data)

    return T1SignalResults(t1s, fitted_parameters)


def _plot(data: T1SignalData, target: QubitId, fit: T1SignalResults = None):
    """Plotting function for T1 experiment."""

    figures = []
    fig = go.Figure()

    fitting_report = None
    qubit_data = data[target]
    waits = qubit_data.wait

    fig.add_trace(
        go.Scatter(
            x=waits,
            y=qubit_data.signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        )
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
            table_dict(target, "T1 [ns]", np.round(fit.t1[target]))
        )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


t1_signal = Routine(_acquisition, _fit, _plot, t1._update)
"""T1 Signal Routine object."""
