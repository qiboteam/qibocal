from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Parameters, Results, Routine
from qibocal.protocols import classification
from qibocal.protocols.utils import table_dict, table_html

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
        ("angle", np.float64),
        ("threshold", np.float64),
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
    best_angles: dict[QubitId, float] = field(default_factory=dict)
    best_thresholds: dict[QubitId, float] = field(default_factory=dict)


def _acquisition(
    params: TwpaPowerParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaPowerData:
    r"""
    Data acquisition for TWPA power optmization.
    This protocol perform a classification protocol for twpa powers
    in the range [twpa_power - power_width / 2, twpa_power + power_width / 2]
    with step power_step.

    Args:
        params (:class:`TwpaPowerParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitId to be characterized

    Returns:
        data (:class:`TwpaFrequencyData`)
    """

    data = TwpaPowerData()

    power_range = np.arange(
        -params.power_width / 2, params.power_width / 2, params.power_step
    )

    initial_twpa_power = {}
    for qubit in targets:
        initial_twpa_power[qubit] = platform.qubits[qubit].twpa.local_oscillator.power
        data.powers[qubit] = list(
            platform.qubits[qubit].twpa.local_oscillator.power + power_range
        )

    for power in power_range:
        for qubit in targets:
            platform.qubits[qubit].twpa.local_oscillator.power = (
                initial_twpa_power[qubit] + power
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
                TwpaPowerType,
                (qubit),
                dict(
                    power=np.array(
                        [float(platform.qubits[qubit].twpa.local_oscillator.power)]
                    ),
                    assignment_fidelity=np.array(
                        [classification_result.assignment_fidelity[qubit]]
                    ),
                    angle=np.array([classification_result.rotation_angle[qubit]]),
                    threshold=np.array([classification_result.threshold[qubit]]),
                ),
            )
    return data


def _fit(data: TwpaPowerData) -> TwpaPowerResults:
    """Extract fidelity for each configuration qubit / param.
    Where param can be either frequency or power."""
    qubits = data.qubits
    best_power = {}
    best_fidelity = {}
    best_angle = {}
    best_threshold = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmax(data_qubit["assignment_fidelity"])
        best_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_err]
        best_power[qubit] = data_qubit["power"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return TwpaPowerResults(
        best_power,
        best_fidelity,
        best_angles=best_angle,
        best_thresholds=best_threshold,
    )


def _plot(data: TwpaPowerData, fit: TwpaPowerResults, target: QubitId):
    """Plotting function that shows the assignment fidelity
    for different values of the twpa power for a single qubit."""

    figures = []
    fitting_report = ""

    if fit is not None:
        qubit_data = data.data[target]
        fidelities = qubit_data["assignment_fidelity"]
        powers = qubit_data["power"]
        fitting_report = table_html(
            table_dict(
                target,
                ["Best assignment fidelity", "TWPA Power [dBm]"],
                [
                    np.round(fit.best_fidelities[target], 3),
                    np.round(fit.best_powers[target], 3),
                ],
            )
        )
        fig = go.Figure([go.Scatter(x=powers, y=fidelities, name="Fidelity")])
        figures.append(fig)

        fig.update_layout(
            showlegend=True,
            xaxis_title="TWPA Power [dB]",
            yaxis_title="Assignment Fidelity",
        )

    return figures, fitting_report


def _update(results: TwpaPowerResults, platform: Platform, target: QubitId):
    update.twpa_power(results.best_powers[target], platform, target)
    update.iq_angle(results.best_angles[target], platform, target)
    update.threshold(results.best_thresholds[target], platform, target)


twpa_power = Routine(_acquisition, _fit, _plot, _update)
"""Twpa power Routine  object."""
