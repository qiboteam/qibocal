from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from sklearn.model_selection import train_test_split

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.fitting.classifier.run import benchmarking
from qibocal.protocols.characterization import classification
from qibocal.protocols.characterization.utils import GHZ_TO_HZ, HZ_TO_GHZ


@dataclass
class TwpaFrequencyParameters(Parameters):
    """Flipping runcard inputs."""

    frequency_width: float
    frequency_step: float

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TwpaFrequencyData(Data):
    """Flipping acquisition outputs."""

    data: dict[
        tuple[QubitId, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""

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

    def __getitem__(self, qubit: QubitId):
        return {
            index: value
            for index, value in self.fidelities.items()
            if index[0] == qubit
        }


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

    data = TwpaFrequencyData()

    initial_twpa_freq = {}
    for qubit in qubits:
        initial_twpa_freq[qubit] = platform.qubits[
            qubit
        ].twpa.local_oscillator.frequency

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

    return TwpaFrequencyResults(fidelities=fidelities)


def _plot(data: TwpaFrequencyData, fit: TwpaFrequencyResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa frequency for a single qubit"""

    figures = []
    fitting_report = "No fitting data"
    qubit_fit = fit[qubit]
    freqs = []
    fidelities = []
    for _, freq in qubit_fit:
        freqs.append(freq * HZ_TO_GHZ)
        fidelities.append(qubit_fit[qubit, freq])

    fitting_report = f"{qubit} | Best assignment fidelity: {np.max(fidelities):.3f}<br>"
    fitting_report += f"{qubit} | TWPA Frequency: {int(freqs[np.argmax(fidelities)]*GHZ_TO_HZ)} Hz <br>"

    fig = go.Figure([go.Scatter(x=freqs, y=fidelities, name="Fidelity")])
    figures.append(fig)

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="TWPA Frequency [GHz]",
        yaxis_title="Assignment Fidelity",
    )

    return figures, fitting_report


twpa_frequency = Routine(_acquisition, _fit, _plot)
"""Twpa frequency Routine  object."""
