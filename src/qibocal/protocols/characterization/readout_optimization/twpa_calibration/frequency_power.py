from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization import classification
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, table_dict, table_html


@dataclass
class TwpaFrequencyPowerParameters(Parameters):
    """Twpa Frequency Power runcard inputs."""

    frequency_width: float
    """Frequency total width."""
    frequency_step: float
    """Frequency step to be probed."""
    power_width: float
    """Power total width."""
    power_step: float
    """Power step to be probed."""


TwpaFrequencyPowerType = np.dtype(
    [
        ("freq", np.float64),
        ("power", np.float64),
        ("assignment_fidelity", np.float64),
    ]
)


@dataclass
class TwpaFrequencyPowerData(Data):
    """Twpa Frequency Power acquisition outputs."""

    data: dict[
        tuple[QubitId, float, float], npt.NDArray[classification.ClassificationType]
    ] = field(default_factory=dict)
    """Raw data acquired."""
    frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Frequencies for each qubit."""
    powers: dict[QubitId, float] = field(default_factory=dict)
    """Powers for each qubit."""


@dataclass
class TwpaFrequencyPowerResults(Results):
    """Twpa Frequency Power outputs."""

    best_freqs: dict[QubitId, float] = field(default_factory=dict)
    best_powers: dict[QubitId, float] = field(default_factory=dict)
    best_fidelities: dict[QubitId, float] = field(default_factory=dict)


def _acquisition(
    params: TwpaFrequencyPowerParameters,
    platform: Platform,
    qubits: Qubits,
) -> TwpaFrequencyPowerData:
    r"""
    Data acquisition for TWPA frequency vs. power optmization.
    This protocol perform a classification protocol for twpa frequencies
    in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2]
    with step frequency_step and powers in the range [twpa_power - power_width / 2, twpa_power + power_width / 2]

    Args:
        params (:class:`TwpaFrequencyPowerParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`TwpaFrequencyPowerData`)
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
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                initial_twpa_freq[qubit] + freq
            )

            for power in power_range:
                for qubit in qubits:
                    platform.qubits[qubit].twpa.local_oscillator.power = (
                        initial_twpa_power[qubit] + power
                    )

                classification_data = classification._acquisition(
                    classification.SingleShotClassificationParameters.load(
                        {"nshots": params.nshots}
                    ),
                    platform,
                    qubits,
                )

                classification_result = classification._fit(classification_data)

                data.register_qubit(
                    TwpaFrequencyPowerType,
                    (qubit),
                    dict(
                        freq=np.array(
                            [platform.qubits[qubit].twpa.local_oscillator.frequency],
                            dtype=np.float64,
                        ),
                        power=np.array(
                            [platform.qubits[qubit].twpa.local_oscillator.power],
                            dtype=np.float64,
                        ),
                        assignment_fidelity=np.array(
                            [classification_result.assignment_fidelity[qubit]],
                        ),
                    ),
                )
    return data


def _fit(data: TwpaFrequencyPowerData) -> TwpaFrequencyPowerResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""

    best_freq = {}
    best_power = {}
    best_fidelity = {}
    qubits = data.qubits

    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmax(data_qubit["assignment_fidelity"])
        best_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_err]
        best_freq[qubit] = data_qubit["freq"][index_best_err]
        best_power[qubit] = data_qubit["power"][index_best_err]

    return TwpaFrequencyPowerResults(best_freq, best_power, best_fidelity)


def _plot(data: TwpaFrequencyPowerData, fit: TwpaFrequencyPowerResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa frequency for a single qubit"""

    figures = []
    fitting_report = ""
    if fit is not None:
        qubit_data = data.data[qubit]
        fidelities = qubit_data["assignment_fidelity"]
        frequencies = qubit_data["freq"]
        powers = qubit_data["power"]
        fitting_report = table_html(
            table_dict(
                qubit,
                ["Best assignment fidelity", "TWPA Frequency [Hz]", "TWPA Power [dBm]"],
                [
                    np.round(fit.best_fidelities[qubit], 3),
                    fit.best_freqs[qubit],
                    np.round(fit.best_powers[qubit], 3),
                ],
            )
        )

        fig = go.Figure(
            [
                go.Heatmap(
                    x=frequencies * HZ_TO_GHZ, y=powers, z=fidelities, name="Fidelity"
                )
            ]
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title="TWPA Frequency [GHz]",
            yaxis_title="TWPA Power [dBm]",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(results: TwpaFrequencyPowerResults, platform: Platform, qubit: QubitId):
    update.twpa_frequency(results.best_freqs[qubit], platform, qubit)
    update.twpa_power(results.best_powers[qubit], platform, qubit)


twpa_frequency_power = Routine(_acquisition, _fit, _plot, _update)
"""Twpa frequency Routine  object."""
