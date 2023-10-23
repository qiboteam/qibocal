from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..utils import table_dict, table_html
from . import utils

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class T1Parameters(Parameters):
    """T1 runcard inputs."""

    delay_before_readout_start: int
    """Initial delay before readout (ns)."""
    delay_before_readout_end: int
    """Final delay before readout (ns)."""
    delay_before_readout_step: int
    """Step delay before readout (ns)."""


@dataclass
class T1Results(Results):
    """T1 outputs."""

    t1: dict[QubitId, tuple[float]]
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


CoherenceProbType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class T1Data(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(params: T1Parameters, platform: Platform, qubits: Qubits) -> T1Data:
    r"""Data acquisition for T1 experiment.
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        params:
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action
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
    for qubit in qubits:
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
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    data = T1Data()

    # sweep the parameter
    # execute the pulse sequence
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    for qubit in qubits:
        probs = results[ro_pulses[qubit].serial].probability(state=1)
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
    t1s, fitted_parameters = utils.exponential_fit_probability(data)
    return T1Results(t1s, fitted_parameters)


def _plot(data: T1Data, qubit, fit: T1Results = None):
    """Plotting function for T1 experiment."""

    figures = []
    fitting_report = ""
    qubit_data = data[qubit]
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
            table_dict(qubit, ["T1 [ns]"], [fit.t1[qubit]], display_error=True)
        )
    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time (ns)",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: T1Results, platform: Platform, qubit: QubitId):
    update.t1(results.t1[qubit], platform, qubit)


t1 = Routine(_acquisition, _fit, _plot, _update)
"""T1 Routine object."""
