from dataclasses import dataclass, field
from os import error

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html


@dataclass
class ReadoutHoldoffParameters(Parameters):
    """Readout holdoff runcard inputs."""

    holdoff_step: int = 4
    """holdoff step to be probed."""
    holdoff_start: int = 100
    """holdoff start."""
    holdoff_stop: float = 600
    """holdoff stop value"""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best holdoff search"""


ReadoutHoldoffType = np.dtype(
    [
        ("error", np.float64),
        ("holdoff", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO acquisition holdoff."""


@dataclass
class ReadoutHoldoffData(Data):
    """Data class for `readout_holdoff` protocol."""

    data: dict[tuple, npt.NDArray[ReadoutHoldoffType]] = field(default_factory=dict)


@dataclass
class ReadoutHoldoffResults(Results):
    """Result class for `readout_holdoff` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_holdoff: dict[QubitId, list]
    """holdoff with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ReadoutHoldoffParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ReadoutHoldoffData:
    r"""
    Data acquisition for readout holdoff optmization.
    This protocol sweeps the readout acquisition holdoff performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.
    
    May only work with QBLOX

    Args:
        params (:class:`ReadoutHoldoffParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ReadoutHoldoffData`)
    """
    from qibolab.channels import Channel
    from qibolab.qubits import Qubit
    from qibolab.instruments.qblox.port import QbloxInputPort
    
    data = ReadoutHoldoffData()
    for qubit_name in targets:
        error = 1
        qubit: Qubit = platform.qubits[qubit_name]

        feddback_channel: Channel = qubit.feedback
        feddback_channel_port: QbloxInputPort = feddback_channel.port
        
        old_holdoff = feddback_channel_port.acquisition_hold_off
        new_holdoff = params.holdoff_start
        while error > params.error_threshold and new_holdoff <= params.holdoff_stop:
            feddback_channel_port.acquisition_holdoff = new_holdoff

            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            qd_pulses = platform.create_RX_pulse(qubit_name, start=0)
            ro_pulses = platform.create_qubit_readout_pulse(
                qubit_name, start=qd_pulses.finish
            )
            sequence_0.add(ro_pulses)
            sequence_1.add(qd_pulses)
            sequence_1.add(ro_pulses)

            state0_results = platform.execute_pulse_sequence(
                sequence_0,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )

            state1_results = platform.execute_pulse_sequence(
                sequence_1,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )
            result0 = state0_results[ro_pulses.serial]
            result1 = state1_results[ro_pulses.serial]

            i_values = np.concatenate((result0.voltage_i, result1.voltage_i))
            q_values = np.concatenate((result0.voltage_q, result1.voltage_q))
            iq_values = np.stack((i_values, q_values), axis=-1)
            nshots = int(len(i_values) / 2)
            states = [0] * nshots + [1] * nshots
            model = QubitFit()
            model.fit(iq_values, np.array(states))
            error = model.probability_error

            data.register_qubit(
                ReadoutHoldoffType,
                (qubit_name),
                dict(
                    holdoff = np.array([new_holdoff]),
                    error=np.array([error]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
            feddback_channel_port.acquisition_hold_off = old_holdoff
            new_holdoff += params.holdoff_step
    return data


def _fit(data: ReadoutHoldoffData) -> ReadoutHoldoffResults:
    qubits = data.qubits
    best_holdoffs = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_holdoffs[qubit] = data_qubit["holdoff"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ReadoutHoldoffResults(lowest_err, best_holdoffs, best_angle, best_threshold)


def _plot(
    data: ReadoutHoldoffData, fit: ReadoutHoldoffResults, target: QubitId
):
    """Plotting function for Optimization RO holdoff."""
    figures = []
    opacity = 1
    fitting_report = None
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data[target]["holdoff"],
                y=data[target]["error"],
                opacity=opacity,
                showlegend=True,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                "Best Readout holdoff [a.u.]",
                np.round(fit.best_holdoff[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout holdoff [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ReadoutHoldoffResults, platform: Platform, target: QubitId):
    update.readout_holdoff(results.best_holdoff[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


readout_holdoff = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Readout Duholdoffration Routine  object."""
