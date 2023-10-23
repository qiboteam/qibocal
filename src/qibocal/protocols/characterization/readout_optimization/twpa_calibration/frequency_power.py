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
from qibocal.protocols.characterization.utils import HZ_TO_GHZ


@dataclass
class TwpaFrequencyPowerParameters(Parameters):
    """Twpa Frequency Power runcard inputs."""

    frequency_width: float
    frequency_step: float
    power_width: float
    """Power total width."""
    power_step: float
    """Power step to be probed."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TwpaFrequencyPowerData(Data):
    """Twpa Frequency Power acquisition outputs."""

    data: dict[
        tuple[QubitId, float, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""

    def register_freq_pow(
        self,
        qubit: QubitId,
        freq: float,
        pow: float,
        classification_data: npt.NDArray[classification.ClassificationType],
    ):
        self.data[qubit, freq, pow] = classification_data[qubit]


@dataclass
class TwpaFrequencyPowerResults(Results):
    """Twpa Frequency Power outputs."""

    fidelities: dict[QubitId, float, float] = field(default_factory=dict)

    def __getitem__(self, qubit: QubitId):
        return {
            index: value
            for index, value in self.fidelities.items()
            if index[0] == qubit
        }


def _acquisition(
    params: TwpaFrequencyPowerParameters,
    platform: Platform,
    qubits: Qubits,
) -> TwpaFrequencyPowerData:
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

    data = TwpaFrequencyPowerData()

    freq_range = np.arange(
        -params.frequency_width / 2, params.frequency_width / 2, params.frequency_step
    ).astype(int)
    power_range = np.arange(
        -params.power_width / 2, params.power_width / 2, params.power_step
    )
    data = TwpaFrequencyPowerData()

    initial_twpa_freq = {}
    initial_twpa_power = {}
    for qubit in qubits:
        initial_twpa_freq[qubit] = platform.qubits[
            qubit
        ].twpa.local_oscillator.frequency
        initial_twpa_power[qubit] = platform.qubits[qubit].twpa.local_oscillator.power

    for freq in freq_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                initial_twpa_freq[qubit] + freq
            )

        for power in power_range:
            for qubit in qubits:
                platform.qubits[qubit].twpa.local_oscillator.power = (
                    initial_twpa_power[qubit] + power
                )

            classification_data = classification._acquisition(
                classification.SingleShotClassificationParameters(nshots=params.nshots),
                platform,
                qubits,
            )

            for qubit in qubits:
                data.register_freq_pow(
                    qubit,
                    platform.qubits[qubit].twpa.local_oscillator.frequency,
                    platform.qubits[qubit].twpa.local_oscillator.power,
                    classification_data,
                )

    return data


def _fit(data: TwpaFrequencyPowerData) -> TwpaFrequencyPowerResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""
    fidelities = {}
    for qubit, freq, pow in data.data:
        qubit_data = data.data[qubit, freq, pow]
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
        fidelities[qubit, freq, pow] = model.assignment_fidelity

    return TwpaFrequencyPowerResults(fidelities=fidelities)


def _plot(data: TwpaFrequencyPowerData, fit: TwpaFrequencyPowerResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa frequency and power for a single qubit"""

    figures = []
    fitting_report = "No fitting data"
    qubit_fit = fit[qubit]
    freqs = []
    pows = []
    fidelities = []
    for _, freq, pow in qubit_fit:
        freqs.append(freq * HZ_TO_GHZ)
        pows.append(pow)
        fidelities.append(qubit_fit[qubit, freq, pow])

    fitting_report = f"{qubit} | Best assignment fidelity: {np.max(fidelities):.3f}<br>"
    # fitting_report += f"{qubit} | TWPA Frequency: {int(freqs[np.argmax(fidelities)]*GHZ_TO_HZ)} Hz <br>"

    fig = go.Figure([go.Heatmap(x=freqs, y=pows, z=fidelities, name="Fidelity")])
    figures.append(fig)

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="TWPA Frequency [GHz]",
        yaxis_title="TWPA Power [dBm]",
    )

    return figures, fitting_report


twpa_frequency_power = Routine(_acquisition, _fit, _plot)
"""Twpa frequency Routine  object."""
