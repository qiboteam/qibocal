import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform

from ..utils import COLORBAND, COLORBAND_LINE

logging.basicConfig(level=logging.INFO)

__all__ = ["jazz"]


JAZZType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
)


@dataclass
class JAZZParameters(Parameters):
    """JAZZ runcard inputs."""

    start_wait: float
    end_wait: float
    nsteps: int


@dataclass
class JAZZResults(Results):
    """JAZZ outputs."""


@dataclass
class JAZZData(Data):
    """JAZZ acquisition outputs."""

    data: dict[QubitPairId, npt.NDArray[JAZZType]] = field(default_factory=dict)


def _acquisition(
    params: JAZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> JAZZData:
    """Data acquisition for JAZZ"""

    if len(targets) != 1:
        raise ValueError("simultaneous pair is not supported for JAZZ.")
    q0 = targets[0][0]
    q1 = targets[0][1]
    wait_range = np.linspace(
        params.start_wait, params.end_wait, params.nsteps, endpoint=True
    )

    q0_drive_channel = platform.qubits[q0].drive
    q0_natives = platform.natives.single_qubit[q0]
    ro_channel, ro_pulse = q0_natives.MZ()[0]
    q1_drive_channel = platform.qubits[q1].drive
    q1_natives = platform.natives.single_qubit[q1]
    delay = Delay(duration=0)

    sequence = PulseSequence()
    # X(pi/2)
    for channel, pulse in q0_natives.R(theta=np.pi / 2):
        sequence += [(channel, pulse)]
        sequence += [(ro_channel, Delay(duration=pulse.duration))]
        sequence += [(q1_drive_channel, Delay(duration=pulse.duration))]
    # delay
    sequence += [(q0_drive_channel, delay)]
    sequence += [(q1_drive_channel, delay)]
    sequence += [(ro_channel, delay)]
    # X(pi)
    for channel, pulse in q0_natives.RX():
        sequence += [(channel, pulse)]
        sequence += [(ro_channel, Delay(duration=pulse.duration))]
        sequence += [(q1_drive_channel, Delay(duration=pulse.duration))]
    # X(pi) on q1
    for channel, pulse in q1_natives.RX():
        sequence += [(channel, pulse)]
        sequence += [(ro_channel, Delay(duration=pulse.duration))]
        sequence += [(q0_drive_channel, Delay(duration=pulse.duration))]
    # delay
    sequence += [(q0_drive_channel, delay)]
    sequence += [(ro_channel, delay)]
    # Y(pi/2)
    for channel, pulse in q0_natives.R(theta=np.pi / 2, phi=np.pi / 2):
        sequence += [(channel, pulse)]
        sequence += [(ro_channel, Delay(duration=pulse.duration))]
    # MZ
    sequence += [(ro_channel, ro_pulse)]

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=wait_range,
        pulses=[delay],
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    prob = results[ro_pulse.id]
    error = np.sqrt(prob * (1 - prob) / params.nshots)
    data = JAZZData()
    data.register_qubit(
        JAZZType,
        (targets[0]),
        dict(
            wait=np.array(wait_range),
            prob=np.array(prob),
            errors=np.array(error),
        ),
    )

    return data


def _plot(data: JAZZData, target: QubitId, fit: JAZZResults = None):
    """Plotting function for JAZZ Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    qubit_data = data.data[target]
    waits = qubit_data["wait"]
    probs = qubit_data["prob"]
    error_bars = qubit_data["errors"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of State 1",
                showlegend=True,
                legendgroup="Probability of State 1",
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

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _fit(data: JAZZData) -> JAZZResults:
    """Post-processing for JAZZ."""
    return JAZZResults()


def _update(*args, **kwargs):
    pass


jazz = Routine(_acquisition, _fit, _plot, _update)
"""JAZZ Routine object."""
