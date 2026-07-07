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

    waits_range: tuple[float, float, float]


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
    # qubit we measure to estimate zz coupling
    target = targets[0][0]
    # qubit we flip in the middle of the sequence
    spectator = targets[0][1]

    # target qubit channels and native pulses
    target_drive_channel = platform.qubits[target].drive
    target_natives = platform.natives.single_qubit[target]
    target_ro_channel, target_ro_pulse = target_natives.MZ()[0]

    # spectator qubit channnels and native pulses
    spectator_drive_channel = platform.qubits[spectator].drive
    spectator_natives = platform.natives.single_qubit[spectator]

    delay = Delay(duration=0)

    sequence = PulseSequence()

    # delay
    sequence += [(target_drive_channel, delay)]
    sequence += [(spectator_drive_channel, delay)]
    sequence += [(target_ro_channel, delay)]

    # X(pi) on both spectator and target qubits
    _, target_pi_pulse = target_natives.RX()[0]
    sequence += [
        (target_drive_channel, target_pi_pulse),
        (target_ro_channel, Delay(duration=target_pi_pulse.duration)),
    ]
    sequence += spectator_natives.RX()

    # delay
    sequence += [(target_drive_channel, delay)]
    sequence += [(target_ro_channel, delay)]

    # Y(pi/2)
    sequence += spectator_natives.R(theta=np.pi / 2, phi=np.pi / 2)

    # measuring target qubit
    sequence |= [(target_ro_channel, target_ro_pulse)]

    # adding the initial X(pi/2) on target
    sequence = target_natives.R(theta=np.pi / 2) | sequence

    sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.waits_range,
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

    prob = results[target_ro_pulse.id]
    error = np.sqrt(prob * (1 - prob) / params.nshots)
    data = JAZZData()
    data.register_qubit(
        JAZZType,
        (targets[0]),
        dict(
            wait=sweeper.values,
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
