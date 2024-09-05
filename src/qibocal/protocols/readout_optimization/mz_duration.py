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
class ReadoutDurationParameters(Parameters):
    """Readout duration runcard inputs."""

    duration_step: int = 4
    """duration step to be probed."""
    duration_start: int = 100
    """duration start."""
    duration_stop: float = 600
    """duration stop value"""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best duration search"""


ReadoutDurationType = np.dtype(
    [
        ("error", np.float64),
        ("duration", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization of MZ duration."""


@dataclass
class ReadoutDurationData(Data):
    """Data class for `readout_duration` protocol."""

    data: dict[tuple, npt.NDArray[ReadoutDurationType]] = field(default_factory=dict)


@dataclass
class ReadoutDurationResults(Results):
    """Result class for `readout_duration` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_duration: dict[QubitId, list]
    """duration with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ReadoutDurationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ReadoutDurationData:
    r"""
    Data acquisition for readout duration optmization.
    This protocol sweeps the readout acquisition duration performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.
    
    May only work with QBLOX

    Args:
        params (:class:`ReadoutDurationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ReadoutDurationData`)
    """
    from qibolab.channels import Channel
    from qibolab.qubits import Qubit
    
    data = ReadoutDurationData()
    for qubit_name in targets:
        error = 1

        from qibolab.native import NativePulse
        pulse_MZ:NativePulse = platform.qubits[qubit_name].native_gates.MZ 
        
        old_duration = pulse_MZ.duration
        new_duration = params.duration_start
        while error > params.error_threshold and new_duration <= params.duration_stop:
            pulse_MZ.duration = new_duration
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
                ReadoutDurationType,
                (qubit_name),
                dict(
                    duration = np.array([new_duration]),
                    error=np.array([error]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
            pulse_MZ.duration = old_duration
            new_duration += params.duration_step
    return data


def _fit(data: ReadoutDurationData) -> ReadoutDurationResults:
    qubits = data.qubits
    best_durations = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_durations[qubit] = data_qubit["duration"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ReadoutDurationResults(lowest_err, best_durations, best_angle, best_threshold)


def _plot(
    data: ReadoutDurationData, fit: ReadoutDurationResults, target: QubitId
):
    """Plotting function for Optimization MZ duration."""
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
                x=data[target]["duration"],
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
                "Best MZ pulse duration [a.u.]",
                np.round(fit.best_duration[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="MZ duration [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ReadoutDurationResults, platform: Platform, target: QubitId):
    update.readout_mz_duration(results.best_duration[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


mz_duration = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Readout MZ duration Routine  object."""
