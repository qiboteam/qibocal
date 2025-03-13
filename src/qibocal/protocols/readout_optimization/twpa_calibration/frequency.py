from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal.auto.operation import Data, Parameters, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols import classification
from qibocal.protocols.utils import HZ_TO_GHZ, table_dict, table_html


@dataclass
class TwpaFrequencyParameters(Parameters):
    """TwpaFrequency runcard inputs."""

    frequency_width: float
    """Relative frequency width [Hz]."""
    frequency_step: float
    """Frequency step [Hz]."""


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
    platform: CalibrationPlatform,
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
    )

    initial_twpa_freq = {}
    for qubit in targets:
        initial_twpa_freq[qubit] = float(
            platform.qubits[qubit].twpa.local_oscillator.frequency
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
