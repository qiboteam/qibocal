from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization import classification
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class ResonatorAmplitudeParameters(Parameters):
    """ResonatorAmplitude runcard inputs."""

    amplitude_width: float
    """Amplitude width to be probed."""
    amplitude_step: float
    """Amplituude step to be probed."""


ResonatorAmplitudeType = np.dtype(
    [
        ("amplitude", np.float64),
        ("assignment_fidelity", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO amplitude."""


@dataclass
class ResonatorAmplitudeData(Data):
    """ResonatorAmplitude acquisition outputs."""

    data: dict[
        tuple[QubitId, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Amplitudes for each qubit."""


@dataclass
class ResonatorAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    best_fidelity: dict[QubitId, float] = field(default_factory=dict)
    best_amplitude: dict[QubitId, float] = field(default_factory=dict)
    best_angle: dict[QubitId, float] = field(default_factory=dict)
    best_threshold: dict[QubitId, float] = field(default_factory=dict)


def _acquisition(
    params: ResonatorAmplitudeParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the readout fidelity at each step.

    Args:
        params (:class:`ResonatorAmplitudeParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`ResonatorAmplitudeData`)
    """

    data = ResonatorAmplitudeData()

    amplitudes = {}
    num_experiments = int(params.amplitude_width // params.amplitude_step)
    for qubit in qubits:
        amplitudes[qubit] = platform.qubits[qubit].native_gates.MZ.amplitude
        data.amplitudes[qubit] = list(
            np.linspace(
                max(0, amplitudes[qubit] - params.amplitude_width / 2),
                min(1, amplitudes[qubit] + params.amplitude_width / 2),
                num_experiments,
            )
        )

    for n in range(num_experiments):
        for qubit in qubits:
            platform.qubits[qubit].native_gates.MZ.amplitude = data.amplitudes[qubit][n]
        classification_data = classification._acquisition(
            classification.SingleShotClassificationParameters.load(
                {"nshots": params.nshots}
            ),
            platform,
            qubits,
        )
        classification_result = classification._fit(classification_data)

        for qubit in qubits:
            data.register_qubit(
                ResonatorAmplitudeType,
                (qubit),
                dict(
                    amplitude=np.array(
                        [data.amplitudes[qubit][n]],
                        dtype=np.float64,
                    ),
                    assignment_fidelity=np.array(
                        [classification_result.assignment_fidelity[qubit]],
                    ),
                    angle=np.array([classification_result.rotation_angle[qubit]]),
                    threshold=np.array([classification_result.threshold[qubit]]),
                ),
            )
    return data


def _fit(data: ResonatorAmplitudeData) -> ResonatorAmplitudeResults:
    qubits = data.qubits
    best_fidelity = {}
    best_amplitude = {}
    best_angle = {}
    best_threshold = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_fidelity = np.argmax(data_qubit["assignment_fidelity"])
        best_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_fidelity]
        best_amplitude[qubit] = data.amplitudes[qubit][index_best_fidelity]
        best_angle[qubit] = data_qubit["angle"][index_best_fidelity]
        best_threshold[qubit] = data_qubit["threshold"][index_best_fidelity]

    return ResonatorAmplitudeResults(
        best_fidelity, best_amplitude, best_angle, best_threshold
    )


def _plot(data: ResonatorAmplitudeData, fit: ResonatorAmplitudeResults, qubit):
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
                x=data[qubit]["amplitude"],
                y=data[qubit]["assignment_fidelity"],
                opacity=opacity,
                showlegend=True,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                qubit,
                "Best Readout Amplitude [a.u.]",
                np.round(fit.best_amplitude[qubit], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude [a.u.]",
        yaxis_title="assignment_fidelity",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorAmplitudeResults, platform: Platform, qubit: QubitId):
    update.readout_amplitude(results.best_amplitude[qubit], platform, qubit)
    update.iq_angle(results.best_angle[qubit], platform, qubit)
    update.threshold(results.best_threshold[qubit], platform, qubit)


resonator_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Amplitude Routine  object."""
