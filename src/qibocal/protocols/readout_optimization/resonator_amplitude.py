from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.classification import _acquisition as cl_acq
from qibocal.protocols.classification import train_classifier
from qibocal.protocols.utils import table_dict, table_html


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
        ("i_values", np.float64),
        ("q_values", np.float64),
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
    for qubit in targets:
        error = 1
        old_amp = platform.qubits[qubit].native_gates.MZ.amplitude
        new_amp = params.amplitude_start
        while error > params.error_threshold and new_amp <= params.amplitude_stop:
            platform.qubits[qubit].native_gates.MZ.amplitude = new_amp
            params.unrolling = False
            cl_data = cl_acq(params, platform, targets)
            model = train_classifier(cl_data, qubit)
            error = model.probability_error
            data.register_qubit(
                ResonatorAmplitudeType,
                (qubit),
                dict(
                    i_values=cl_data.data[qubit]["i"],
                    q_values=cl_data.data[qubit]["q"],
                    amp=np.array([new_amp]),
                    error=np.array([error]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
            platform.qubits[qubit].native_gates.MZ.amplitude = old_amp
            new_amp += params.amplitude_step
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
