from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Qubits, Routine
from qibocal.protocols.characterization import classification
from qibocal.protocols.characterization.utils import table_dict, table_html

from . import frequency


@dataclass
class TwpaPowerParameters(Parameters):
    """TwpaPower runcard inputs."""

    power_width: float
    """Power total width."""
    power_step: float
    """Power step to be probed."""


@dataclass
class TwpaPowerData(frequency.TwpaFrequencyData):
    """Data class for twpa power protocol."""

    powers: dict[QubitId, float] = field(default_factory=dict)
    """Frequencies for each qubit."""


@dataclass
class TwpaPowerResults(frequency.TwpaFrequencyResults):
    """Result class for twpa power protocol."""


def _acquisition(
    params: TwpaPowerParameters,
    platform: Platform,
    qubits: Qubits,
) -> TwpaPowerData:
    r"""
    Data acquisition for TWPA power optmization.
    This protocol perform a classification protocol for twpa powers
    in the range [twpa_power - power_width / 2, twpa_power + power_width / 2]
    with step power_step.

    Args:
        params (:class:`TwpaPowerParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`TwpaFrequencyData`)
    """

    data = TwpaPowerData()

    power_range = np.arange(
        -params.power_width / 2, params.power_width / 2, params.power_step
    )

    initial_twpa_power = {}
    for qubit in qubits:
        initial_twpa_power[qubit] = platform.qubits[qubit].twpa.local_oscillator.power
        data.powers[qubit] = list(
            platform.qubits[qubit].twpa.local_oscillator.power + power_range
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

        for qubit in qubits:
            data.register_freq(
                qubit,
                platform.qubits[qubit].twpa.local_oscillator.power,
                classification_data,
            )

    return data


def _plot(data: TwpaPowerData, fit: TwpaPowerResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa power for a single qubit."""

    figures = []
    fitting_report = ""

    if fit is not None:
        fidelities = []
        powers = np.array(data.powers[qubit])
        for qubit_id, power in fit.fidelities:
            if qubit_id == qubit:
                fidelities.append(fit.fidelities[qubit, power])
        fitting_report = table_html(
            table_dict(
                qubit,
                ["Best assignment fidelity", "TWPA Power"],
                [
                    np.round(np.max(fidelities), 3),
                    np.round(powers[np.argmax(fidelities)], 3),
                ],
            )
        )
        fig = go.Figure([go.Scatter(x=powers, y=fidelities, name="Fidelity")])
        figures.append(fig)

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="TWPA Power [dB]",
            yaxis_title="Assignment Fidelity",
        )

    return figures, fitting_report


twpa_power = Routine(_acquisition, frequency._fit, _plot)
"""Twpa power Routine  object."""
