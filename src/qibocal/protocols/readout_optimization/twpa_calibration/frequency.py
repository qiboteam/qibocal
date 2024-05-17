from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols import classification
from qibocal.protocols.readout_optimization.resonator_frequency import (
    ResonatorFrequencyType,
)
from qibocal.protocols.utils import HZ_TO_GHZ, table_dict, table_html


@dataclass
class TwpaFrequencyParameters(Parameters):
    """TwpaFrequency runcard inputs."""

    frequency_width: float
    """Relative frequency width [Hz]"""
    frequency_step: float
    """Frequency step [Hz]"""


@dataclass
class TwpaFrequencyData(Data):
    """TwpaFrequency acquisition outputs."""

    data: dict[
        tuple[QubitId, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""
    frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Frequencies for each qubit."""


@dataclass
class TwpaFrequencyResults(Results):
    """TwpaFrequency outputs."""

    best_freqs: dict[QubitId, float] = field(default_factory=dict)
    best_fidelities: dict[QubitId, float] = field(default_factory=dict)
    best_angles: dict[QubitId, float] = field(default_factory=dict)
    best_thresholds: dict[QubitId, float] = field(default_factory=dict)


def _acquisition(
    params: TwpaFrequencyParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaFrequencyData:
    r"""
    Data acquisition for TWPA power optmization.
    This protocol perform a classification protocol for twpa frequencies
    in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2]
    with step frequency_step.

    Args:
        params (:class:`TwpaFrequencyParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`TwpaFrequencyData`)
    """

    data = TwpaFrequencyData()

    freq_range = np.arange(
        -params.frequency_width / 2, params.frequency_width / 2, params.frequency_step
    ).astype(int)

    initial_twpa_freq = {}
    for qubit in targets:
        initial_twpa_freq[qubit] = float(
            platform.qubits[qubit].twpa.local_oscillator.frequency
        )
        data.frequencies[qubit] = list(
            float(platform.qubits[qubit].twpa.local_oscillator.frequency) + freq_range
        )

    for freq in freq_range:
        for qubit in targets:
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                initial_twpa_freq[qubit] + freq
            )

        classification_data = classification._acquisition(
            classification.SingleShotClassificationParameters.load(
                {"nshots": params.nshots}
            ),
            platform,
            targets,
        )
        classification_result = classification._fit(classification_data)
        for qubit in targets:
            data.register_qubit(
                ResonatorFrequencyType,
                (qubit),
                dict(
                    freq=np.array(
                        [platform.qubits[qubit].twpa.local_oscillator.frequency],
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


def _fit(data: TwpaFrequencyData) -> TwpaFrequencyResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""

    qubits = data.qubits
    best_freq = {}
    best_fidelity = {}
    best_angle = {}
    best_threshold = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmax(data_qubit["assignment_fidelity"])
        best_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_err]
        best_freq[qubit] = data_qubit["freq"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return TwpaFrequencyResults(
        best_freq, best_fidelity, best_thresholds=best_threshold, best_angles=best_angle
    )


def _plot(data: TwpaFrequencyData, fit: TwpaFrequencyResults, target: QubitId):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa frequency for a single qubit"""

    figures = []
    fitting_report = ""
    if fit is not None:
        qubit_data = data.data[target]
        fidelities = qubit_data["assignment_fidelity"]
        frequencies = qubit_data["freq"]
        fitting_report = table_html(
            table_dict(
                target,
                ["Best assignment fidelity", "TWPA Frequency [Hz]"],
                [
                    np.round(fit.best_fidelities[target], 3),
                    fit.best_freqs[target],
                ],
            )
        )
        fig = go.Figure(
            [go.Scatter(x=frequencies * HZ_TO_GHZ, y=fidelities, name="Fidelity")]
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title="TWPA Frequency [GHz]",
            yaxis_title="Assignment Fidelity",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(results: TwpaFrequencyResults, platform: Platform, target: QubitId):
    update.twpa_frequency(results.best_freqs[target], platform, target)
    update.iq_angle(results.best_angles[target], platform, target)
    update.threshold(results.best_thresholds[target], platform, target)


twpa_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Twpa frequency Routine  object."""
