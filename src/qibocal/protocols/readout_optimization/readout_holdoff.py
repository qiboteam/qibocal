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
from qibolab.channels import Channel
from qibolab.qubits import Qubit

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

    from qibolab.instruments.qblox.port import QbloxInputPort
   
    data = ReadoutHoldoffData()
   
    error = 1
    ports, old_holdoff, = {}, {}
    new_holdoff = params.holdoff_start
    
    # make a dictionnary with the qubits assicated to each port
    for qubit_name in targets:
        qubit = platform.qubits[qubit_name]
        feedback_channel: Channel = qubit.feedback
        port_name = feedback_channel.name + '_' + feedback_channel.port.name

        if not isinstance(feedback_channel.port, QbloxInputPort):
            raise TypeError(
                f"Port {feedback_channel.port.name} is not a QbloxInputPort."
            )
        
        if port_name not in ports:
            ports[port_name] = {'port': feedback_channel.port, 'qubits' : []}
            old_holdoff[port_name] = ports[port_name]['port'].acquisition_hold_off
        ports[port_name]['qubits'].append(qubit_name)

    while error > params.error_threshold and new_holdoff <= params.holdoff_stop:
        for port_name, port_info in ports.items():
            port: QbloxInputPort = port_info['port']
            target_qubits = port_info['qubits']
            port.acquisition_hold_off = new_holdoff

            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            ro_pulses, qd_pulses = {}, {}
            for qubit in target_qubits:
                qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit, start=qd_pulses[qubit].finish
                )
            
            for qubit in target_qubits:
                sequence_0.add(ro_pulses[qubit]) 
                sequence_1.add(qd_pulses[qubit])
                sequence_1.add(ro_pulses[qubit])

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
        
            # Gather results
            for qubit in target_qubits:
                result0 = state0_results[ro_pulses[qubit].serial]
                result1 = state1_results[ro_pulses[qubit].serial]

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
                    (qubit),
                    dict(
                        holdoff = np.array([new_holdoff]),
                        error=np.array([error]),
                        angle=np.array([model.angle]),
                        threshold=np.array([model.threshold]),
                    ),
                )
            port.acquisition_hold_off = old_holdoff[port_name]
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
