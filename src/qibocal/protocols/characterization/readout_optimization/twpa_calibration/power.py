from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Parameters, Qubits, Results, Routine
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


TwpaPowerType = np.dtype(
    [
        ("power", np.float64),
        ("assignment_fidelity", np.float64),
    ]
)


@dataclass
class TwpaPowerData(frequency.TwpaFrequencyData):
    """Data class for twpa power protocol."""

    powers: dict[QubitId, float] = field(default_factory=dict)
    """Frequencies for each qubit."""


@dataclass
class TwpaPowerResults(Results):
    """Result class for twpa power protocol."""

    best_powers: dict[QubitId, float] = field(default_factory=dict)
    best_fidelities: dict[QubitId, float] = field(default_factory=dict)


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
        classification_result = classification._fit(classification_data)

        for qubit in qubits:
            data.register_qubit(
                TwpaPowerType,
                (qubit),
                dict(
                    power=np.array(
                        [float(platform.qubits[qubit].twpa.local_oscillator.power)]
                    ),
                    assignment_fidelity=np.array(
                        [classification_result.assignment_fidelity[qubit]]
                    ),
                ),
            )
    return data


def _fit(data: TwpaPowerData) -> TwpaPowerResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""
    qubits = data.qubits
    best_power = {}
    best_fidelity = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmax(data_qubit["assignment_fidelity"])
        best_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_err]
        best_power[qubit] = data_qubit["power"][index_best_err]

    return TwpaPowerResults(best_power, best_fidelity)


def _plot(data: TwpaPowerData, fit: TwpaPowerResults, qubit):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa power for a single qubit."""

    figures = []
    fitting_report = ""

    if fit is not None:
        qubit_data = data.data[qubit]
        fidelities = qubit_data["assignment_fidelity"]
        powers = qubit_data["power"]
        fitting_report = table_html(
            table_dict(
                qubit,
                ["Best assignment fidelity", "TWPA Power"],
                [
                    np.round(fit.best_fidelities[qubit], 3),
                    np.round(fit.best_powers[qubit], 3),
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


def _update(results: TwpaPowerResults, platform: Platform, qubit: QubitId):
    update.twpa_power(results.best_powers[qubit], platform, qubit)


twpa_power = Routine(_acquisition, _fit, _plot, _update)
"""Twpa power Routine  object."""
