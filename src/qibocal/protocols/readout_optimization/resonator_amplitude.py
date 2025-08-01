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
from qibolab.sweeper import Parameter, Sweeper, SweeperType


@dataclass
class ResonatorAmplitudeParameters(Parameters):
    """ResonatorAmplitude runcard inputs."""

    amplitude_step: float
    """Amplituude step to be probed."""
    amplitude_start: float = 0.0
    """Amplitude start."""
    amplitude_stop: float = 1.0
    """Amplitude stop value"""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best amplitude search"""


ResonatorAmplitudeType = np.dtype(
    [
        ("error", np.float64),
        ("amp", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO amplitude."""


@dataclass
class ResonatorAmplitudeData(Data):
    """Data class for `resoantor_amplitude` protocol."""

    data: dict[tuple, npt.NDArray[ResonatorAmplitudeType]] = field(default_factory=dict)


@dataclass
class ResonatorAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_amp: dict[QubitId, list]
    """Amplitude with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ResonatorAmplitudeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ResonatorAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ResonatorAmplitudeParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ResonatorAmplitudeData`)
    """

    data = ResonatorAmplitudeData()
    if True:
        error = 1
        
        amplitude_range = np.arange(
            params.amplitude_start, params.amplitude_stop, params.amplitude_step
        )
 
        sequence_0 = PulseSequence()
        sequence_1 = PulseSequence()

        qd_pulses, ro_pulses = {},{}
        for qubit in targets:
            qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)

            sequence_0.add(ro_pulses[qubit])
            sequence_1.add(qd_pulses[qubit])
            sequence_1.add(ro_pulses[qubit])

        sweeper = Sweeper(
            Parameter.amplitude,
            amplitude_range,
            pulses=[ro_pulses[qubit] for qubit in targets],
            type=SweeperType.ABSOLUTE,
        )

        results_0 = platform.sweep(
            sequence_0,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            ),
            sweeper,
        )
        
        results_1 = platform.sweep(
            sequence_1,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            ),
            sweeper,
        )

        for qubit in targets:
            for k, amp in enumerate(delta_amplitude_range := amplitude_range):
                i_values = []
                q_values = []
                states = []
                for i, results in enumerate([results_0, results_1]):
                    result = results[ro_pulses[qubit].serial]
                    i_values.extend(result.voltage_i[:, k])
                    q_values.extend(result.voltage_q[:, k])
                    states.extend([i] * len(result.voltage_i[:, k]))

                model = QubitFit()
                model.fit(np.stack((i_values, q_values), axis=-1), np.array(states))
                error = model.probability_error
                data.register_qubit(
                    ResonatorAmplitudeType,
                    (qubit),
                    dict(
                        amp=np.array([amp]),
                        error=np.array([error]),
                        angle=np.array([model.angle]),
                        threshold=np.array([model.threshold]),
                    ),
                )

    return data


def _fit(data: ResonatorAmplitudeData) -> ResonatorAmplitudeResults:
    qubits = data.qubits
    best_amps = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_amps[qubit] = data_qubit["amp"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ResonatorAmplitudeResults(lowest_err, best_amps, best_angle, best_threshold)


def _plot(
    data: ResonatorAmplitudeData, fit: ResonatorAmplitudeResults, target: QubitId
):
    """Plotting function for Optimization RO amplitude."""
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
                x=data[target]["amp"],
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
                "Best Readout Amplitude [a.u.]",
                np.round(fit.best_amp[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorAmplitudeResults, platform: Platform, target: QubitId):
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Amplitude Routine  object."""
