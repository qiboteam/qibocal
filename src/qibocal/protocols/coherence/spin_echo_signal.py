from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Parameters, Results, Routine

from ..utils import table_dict, table_html
from .t1_signal import T1SignalData
from .utils import CoherenceType, exp_decay, exponential_fit


@dataclass
class SpinEchoSignalParameters(Parameters):
    """SpinEcho Signal runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between pulses [ns]."""
    delay_between_pulses_end: int
    """Final delay between pulses [ns]."""
    delay_between_pulses_step: int
    """Step delay between pulses [ns]."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""
    single_shot: bool = False
    """If ``True`` save single shot signal data."""


@dataclass
class SpinEchoSignalResults(Results):
    """SpinEchoSignal outputs."""

    t2_spin_echo: dict[QubitId, Union[float, list[float]]]
    """T2 echo for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    pcov: dict[QubitId, list[float]]
    """Approximate covariance of fitted parameters."""


class SpinEchoSignalData(T1SignalData):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoSignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> SpinEchoSignalData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    ro_pulses = {}
    RX90_pulses1 = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in targets:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX_pulses[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=(
            AveragingMode.SINGLESHOT if params.single_shot else AveragingMode.CYCLIC
        ),
    )

    sequences, all_ro_pulses = [], []

    # sweep the parameter
    for wait in ro_wait_range:
        # save data as often as defined by points

        for qubit in targets:
            RX_pulses[qubit].start = RX90_pulses1[qubit].finish + wait / 2
            RX90_pulses2[qubit].start = RX_pulses[qubit].finish + wait / 2
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish

        sequences.append(deepcopy(sequence))
        all_ro_pulses.append(deepcopy(sequence).ro_pulses)

    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)

    elif not params.unrolling:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    data = SpinEchoSignalData()
    for ig, (wait, ro_pulses) in enumerate(zip(ro_wait_range, all_ro_pulses)):
        for qubit in targets:
            serial = ro_pulses.get_qubit_pulses(qubit)[0].serial
            if params.unrolling:
                result = results[serial][0]
            else:
                result = results[ig][serial]
            if params.single_shot:
                _wait = np.array(len(result.magnitude) * [wait])
            else:
                _wait = np.array([wait])
            data.register_qubit(
                CoherenceType,
                (qubit),
                dict(
                    wait=_wait,
                    signal=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )

    if params.single_shot:
        data.data = {
            qubit: values.reshape((len(ro_wait_range), params.nshots)).T
            for qubit, values in data.data.items()
        }

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
                [np.round(fit.t2_spin_echo[target])],
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


def _update(results: SpinEchoSignalResults, platform: Platform, target: QubitId):
    update.t2_spin_echo(results.t2_spin_echo[target], platform, target)


spin_echo_signal = Routine(_acquisition, _fit, _plot, _update)
"""SpinEcho Routine object."""
