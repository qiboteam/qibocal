from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase

from ... import update
from ..utils import readout_frequency, table_dict, table_html
from . import utils

__all__ = [
    "t1_signal",
    "T1SignalData",
    "T1SignalParameters",
    "T1SignalResults",
    "t1_sequence",
    "update_t1",
]


@dataclass
class T1SignalParameters(Parameters):
    """T1 runcard inputs."""

    delay_before_readout_start: int
    """Initial delay before readout [ns]."""
    delay_before_readout_end: int
    """Final delay before readout [ns]."""
    delay_before_readout_step: int
    """Step delay before readout [ns]."""
    single_shot: bool = False
    """If ``True`` save single shot signal data."""


@dataclass
class T1SignalResults(Results):
    """T1 Signal outputs."""

    t1: dict[QubitId, Union[float, list[float]]]
    """T1 for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""


@dataclass
class T1SignalData(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def average(self):
        if len(next(iter(self.data.values())).shape) > 1:
            return utils.average_single_shots(self.__class__, self.data)
        return self


def t1_sequence(
    platform: CalibrationPlatform,
    targets: list[QubitId],
    flux_pulse_amplitude: Optional[float] = None,
):
    """Create sequence for T1 experiment with a given optional delay."""
    sequence = PulseSequence()
    ro_pulses = {}
    delays = len(targets) * [Delay(duration=0)]
    for i, q in enumerate(targets):
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        ro_pulses[q] = ro_pulse
        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        if flux_pulse_amplitude is not None:
            flux_pulses = len(targets) * [
                Pulse(
                    duration=0, amplitude=flux_pulse_amplitude, envelope=Rectangular()
                )
            ]
            flux_channel = platform.qubits[q].flux
            sequence.append((flux_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((flux_channel, flux_pulses[i]))
        else:
            flux_pulses = []
        sequence.append((ro_channel, delays[i]))
        sequence.append((ro_channel, ro_pulse))

    return sequence, ro_pulses, delays + flux_pulses


def _acquisition(
    params: T1SignalParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> T1SignalData:
    """Data acquisition for T1 experiment.

    In this protocol the y axis is the magnitude of signal in the IQ plane."""

    sequence, ro_pulses, pulses = t1_sequence(platform, targets)

    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=ro_wait_range,
        pulses=pulses,
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

    data = T1SignalData()

    for q in targets:
        result = results[ro_pulses[q].id]
        signal = magnitude(result)
        if params.single_shot:
            _waits = np.array(len(signal) * [ro_wait_range])
        else:
            _waits = ro_wait_range
        data.register_qubit(
            utils.CoherenceType,
            (q),
            dict(wait=_waits, signal=signal, phase=phase(result)),
        )

    return data


def _fit(data: T1SignalData) -> T1SignalResults:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.
    """
    data = data.average
    t1s, fitted_parameters, pcovs = utils.exponential_fit(data)

    return T1SignalResults(t1s, fitted_parameters, pcovs)


def _plot(data: T1SignalData, target: QubitId, fit: T1SignalResults = None):
    """Plotting function for T1 experiment."""
    data = data.average

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
            table_dict(
                target, ["T1 [ns]"], [np.round(fit.t1[target])], display_error=True
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def update_t1(results: T1SignalResults, platform: CalibrationPlatform, target: QubitId):
    update.t1(results.t1[target], platform, target)


t1_signal = Routine(_acquisition, _fit, _plot, update_t1)
"""T1 Signal Routine object."""
