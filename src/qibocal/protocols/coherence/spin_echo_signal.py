from dataclasses import dataclass
from typing import Union

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase

from ... import update
from ..utils import readout_frequency, table_dict, table_html
from .t1_signal import T1SignalData
from .utils import (
    CoherenceType,
    dynamical_decoupling_sequence,
    exp_decay,
    exponential_fit,
)

__all__ = [
    "SpinEchoSignalParameters",
    "SpinEchoSignalResults",
    "spin_echo_signal",
    "update_spin_echo",
]


@dataclass
class SpinEchoSignalParameters(Parameters):
    """SpinEcho Signal runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between pulses [ns]."""
    delay_between_pulses_end: int
    """Final delay between pulses [ns]."""
    delay_between_pulses_step: int
    """Step delay between pulses [ns]."""
    single_shot: bool = False


@dataclass
class SpinEchoSignalResults(Results):
    """SpinEchoSignal outputs."""

    t2: dict[QubitId, Union[float, list[float]]]
    """T2 echo for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""


class SpinEchoSignalData(T1SignalData):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoSignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SpinEchoSignalData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    sequence, delays = dynamical_decoupling_sequence(platform, targets, kind="CP")

    # define the parameter to sweep and its range:
    # delay between pulses
    wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    durations = []
    for q in targets:
        # this is assuming that RX and RX90 have the same duration
        duration = platform.natives.single_qubit[q].RX().duration
        durations.append(duration)
        assert (params.delay_between_pulses_start - duration) / 2 >= 0, (
            f"Initial delay too short for qubit {q}, minimum delay should be {duration}"
        )

    assert len(set(durations)) == 1, (
        "Cannot run on mulitple qubit with different RX duration."
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=(wait_range - durations[0]) / 2,
        pulses=delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=(
            AveragingMode.SINGLESHOT if params.single_shot else AveragingMode.CYCLIC
        ),
    )

    data = SpinEchoSignalData()
    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        result = results[ro_pulse.id]
        signal = magnitude(result)
        if params.single_shot:
            _wait = np.array(len(signal) * [wait_range])
        else:
            _wait = wait_range
        data.register_qubit(
            CoherenceType,
            (qubit),
            dict(
                wait=_wait,
                signal=signal,
                phase=phase(result),
            ),
        )

    return data


def _fit(data: SpinEchoSignalData) -> SpinEchoSignalResults:
    """Post-processing for SpinEcho."""
    data = data.average

    t2echos, fitted_parameters, pcov = exponential_fit(data)

    return SpinEchoSignalResults(t2echos, fitted_parameters, pcov)


def _plot(data: SpinEchoSignalData, target: QubitId, fit: SpinEchoSignalResults = None):
    """Plotting for SpinEcho"""
    data = data.average

    figures = []
    fig = go.Figure()

    # iterate over multiple data folders
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
        ),
    )

    if fit is not None:
        # add fitting trace
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(qubit_data),
        )
        params = fit.fitted_parameters[target]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )

        fitting_report = table_html(
            table_dict(
                target,
                ["T2 Spin Echo [ns]"],
                [np.round(fit.t2[target])],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def update_spin_echo(
    results: SpinEchoSignalResults, platform: CalibrationPlatform, target: QubitId
):
    update.t2_spin_echo(results.t2[target], platform, target)


spin_echo_signal = Routine(_acquisition, _fit, _plot, update_spin_echo)
"""SpinEcho Routine object."""
