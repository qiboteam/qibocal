from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform

from qibocal.auto.operation import Parameters, Qubits, Routine
from qibocal.protocols.characterization import classification

from . import frequency


@dataclass
class TwpaPowerParameters(Parameters):
    """TwpaPower runcard inputs."""

    power_width: float
    """Power total width."""
    power_step: float
    """Power step to be probed."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TwpaPowerData(frequency.TwpaFrequencyData):
    """Data class for twpa power protocol."""


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
    ).astype(int)

    data = TwpaPowerData()

    initial_twpa_power = {}
    for qubit in qubits:
        initial_twpa_power[qubit] = platform.qubits[qubit].twpa.local_oscillator.power

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
    fitting_report = "No fitting data"
    qubit_fit = fit[qubit]
    powers = []
    fidelities = []
    for _, power in qubit_fit:
        powers.append(power)
        fidelities.append(qubit_fit[qubit, power])

    fitting_report = f"{qubit} | Best assignment fidelity: {np.max(fidelities):.3f}<br>"
    fitting_report += (
        f"{qubit} | TWPA power: {powers[np.argmax(fidelities)]:.3f} dB <br>"
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
