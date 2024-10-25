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
class ResonatorDurationParameters(Parameters):
    """ResonatorDuration runcard inputs."""

    duration_step: int = 100
    """Amplituude step to be probed."""
    duration_start: int = 8
    """Duration start."""
    duration_stop: int = 5000
    """Duration stop value"""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best duration search"""


ResonatorDurationType = np.dtype(
    [
        ("error", np.float64),
        ("dur", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO duration."""


@dataclass
class ResonatorDurationData(Data):
    """Data class for `resoantor_duration` protocol."""

    data: dict[tuple, npt.NDArray[ResonatorDurationType]] = field(default_factory=dict)


@dataclass
class ResonatorDurationResults(Results):
    """Result class for `resonator_duration` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_dur: dict[QubitId, list]
    """Duration with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ResonatorDurationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ResonatorDurationData:
    r"""
    Data acquisition for resoantor duration optmization.
    This protocol sweeps the readout duration performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ResonatorDurationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ResonatorDurationData`)
    """

    data = ResonatorDurationData()
    for qubit in targets:
        error = 1
        old_dur = platform.qubits[qubit].native_gates.MZ.duration
        new_dur = params.duration_start
        while error > params.error_threshold and new_dur <= params.duration_stop:
            platform.qubits[qubit].native_gates.MZ.duration = new_dur
            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            qd_pulses = platform.create_RX_pulse(qubit, start=0)
            ro_pulses = platform.create_qubit_readout_pulse(
                qubit, start=qd_pulses.finish
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
                ResonatorDurationType,
                (qubit),
                dict(
                    dur=np.array([new_dur]),
                    error=np.array([error]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
            print(qubit, new_dur, error)
            platform.qubits[qubit].native_gates.MZ.duration = old_dur
            new_dur += params.duration_step
    return data


def _fit(data: ResonatorDurationData) -> ResonatorDurationResults:
    qubits = data.qubits
    best_durs = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_durs[qubit] = data_qubit["dur"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ResonatorDurationResults(lowest_err, best_durs, best_angle, best_threshold)


def _plot(
    data: ResonatorDurationData, fit: ResonatorDurationResults, target: QubitId
):
    """Plotting function for Optimization RO duration."""
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
                x=data[target]["dur"],
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
                "Best Readout Duration [a.u.]",
                np.round(fit.best_dur[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Duration [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorDurationResults, platform: Platform, target: QubitId):
    # update.readout_duration(results.best_dur[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_duration = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Duration Routine  object."""
