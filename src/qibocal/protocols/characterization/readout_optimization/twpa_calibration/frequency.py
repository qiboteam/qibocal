from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from sklearn.model_selection import train_test_split

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.fitting.classifier.run import benchmarking
from qibocal.protocols.characterization import classification
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, table_dict, table_html


@dataclass
class TwpaFrequencyParameters(Parameters):
    """TwpaFrequency runcard inputs."""

    frequency_width: float
    """Relative frequency width [Hz]"""
    frequency_step: float
    """Frequency step [Hz]"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TwpaFrequencyData(Data):
    """TwpaFrequency acquisition outputs."""

    data: dict[
        tuple[QubitId, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""
    frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Frequencies for each qubit."""

    def register_freq(
        self,
        qubit: QubitId,
        freq: float,
        classification_data: npt.NDArray[classification.ClassificationType],
    ):
        self.data[qubit, freq] = classification_data[qubit]


@dataclass
class TwpaFrequencyResults(Results):
    """TwpaFrequency outputs."""

    fidelities: dict[QubitId, float] = field(default_factory=dict)
    best: dict[QubitId, float] = field(default_factory=dict)


def _acquisition(
    params: TwpaFrequencyParameters,
    platform: Platform,
    qubits: Qubits,
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
    for qubit in qubits:
        initial_twpa_freq[qubit] = platform.qubits[
            qubit
        ].twpa.local_oscillator.frequency
        data.frequencies[qubit] = list(
            platform.qubits[qubit].twpa.local_oscillator.frequency + freq_range
        )

    for freq in freq_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                initial_twpa_freq[qubit] + freq
            )

        classification_data = classification._acquisition(
            classification.SingleShotClassificationParameters(nshots=params.nshots),
            platform,
            qubits,
        )

        for qubit in qubits:
            data.register_freq(
                qubit,
                platform.qubits[qubit].twpa.local_oscillator.frequency,
                classification_data,
            )

    return data


def _fit(data: TwpaFrequencyData) -> TwpaFrequencyResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""
    fidelities = {}
    best_param = {}
    for qubit, param in data.data:
        qubit_data = data.data[qubit, param]
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(qubit_data[["i", "q"]].tolist())[:, :],
            np.array(qubit_data[["state"]].tolist())[:, 0],
            test_size=0.25,
            random_state=0,
            shuffle=True,
        )

        model = QubitFit()
        results, y_pred, model, fit_info = benchmarking(
            model, x_train, y_train, x_test, y_test
        )
        fidelities[qubit, param] = model.assignment_fidelity

    for qubit, _ in data.data:
        qubit_fidelities = {
            key: fidelity for key, fidelity in fidelities.items() if key[0] == qubit
        }
        best_param[qubit] = max(qubit_fidelities, key=qubit_fidelities.get)[1]

    return TwpaFrequencyResults(fidelities=fidelities, best=best_param)


def _plot(data: TwpaFrequencyData, fit: TwpaFrequencyResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa frequency for a single qubit"""

    figures = []
    fitting_report = ""
    if fit is not None:
        fidelities = []
        frequencies = np.array(data.frequencies[qubit])
        for qubit_id, freq in fit.fidelities:
            if qubit == qubit_id:
                fidelities.append(fit.fidelities[qubit, freq])
        fitting_report = table_html(
            table_dict(
                qubit,
                ["Best assignment fidelity", "TWPA Frequency [Hz]"],
                [
                    np.round(np.max(fidelities), 3),
                    np.round(fit.best[qubit], 3),
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


def _update(results: TwpaFrequencyResults, platform: Platform, qubit: QubitId):
    update.twpa_frequency(results.best[qubit], platform, qubit)


twpa_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Twpa frequency Routine  object."""
