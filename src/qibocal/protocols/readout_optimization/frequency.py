from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html

__all__ = ["readout_frequency_optimization"]


@dataclass
class ReadoutFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""


@dataclass
class ReadoutFrequencyResults(Results):
    """Optimization RO frequency results."""

    assignment_fidelity: dict[QubitId, list]
    """Assignment fidelities."""
    angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""
    frequency: dict[QubitId, float]
    """Best frequency."""
    index: dict[QubitId, int]
    """Index corresponding to best fidelity."""


@dataclass
class ReadoutFrequencyData(Data):
    """Optimization RO frequency acquisition outputs."""

    nshots: int
    frequencies: dict[QubitId, list[float]] = field(default_factory=dict)
    data: dict[tuple[QubitId, int], np.ndarray] = field(default_factory=dict)


def _acquisition(
    params: ReadoutFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutFrequencyData:
    r"""
    Data acquisition for readout frequency optimization.
    While sweeping the readout frequency, the routine performs a single shot
    classification and evaluates the assignment fidelity.
    At the end, the readout frequency is updated, choosing the one that has
    the highest assignment fidelity.

    Args:
        params (ReadoutFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action

    """

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse_0 = natives.MZ()[0]
        _, ro_pulse_1 = natives.MZ()[0]

        sequence_0.append((ro_channel, ro_pulse_0))

        sequence_1.append((qd_channel, qd_pulse))
        sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence_1.append((ro_channel, ro_pulse_1))

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    data = ReadoutFrequencyData(
        nshots=params.nshots,
        frequencies={
            qubit: sweeper.values.tolist() for qubit, sweeper in zip(targets, sweepers)
        },
    )

    state0_results = platform.execute(
        [sequence_0],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    state1_results = platform.execute(
        [sequence_1],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    for state, sequence, results in zip(
        [0, 1], [sequence_0, sequence_1], [state0_results, state1_results]
    ):
        for q in targets:
            data.data[q, state] = results[
                list(sequence.channel(platform.qubits[q].acquisition))[-1].id
            ]

    return data


def _fit(data: ReadoutFrequencyData) -> ReadoutFrequencyResults:
    """Post-Processing for Optimization RO frequency"""

    angle = {qubit: [] for qubit in data.qubits}
    threshold = {qubit: [] for qubit in data.qubits}
    assignment_fidelity = {qubit: [] for qubit in data.qubits}
    frequency = {}
    nshots = data.nshots
    y = np.array([0] * nshots + [1] * nshots)
    index = {}
    for qubit in data.qubits:
        for i in range(len(data.frequencies[qubit])):
            X = np.vstack([data.data[qubit, 0][:, i, :], data.data[qubit, 1][:, i, :]])
            lda = LinearDiscriminantAnalysis().fit(X, y)
            w = lda.coef_[0]
            b = lda.intercept_[0]

            angle[qubit].append(-np.arctan2(w[1], w[0]))
            threshold[qubit].append(-b / np.linalg.norm(w))

            assignment_fidelity[qubit].append(
                np.array(y == lda.predict(X)).sum() / 2 / nshots
            )

        index[qubit] = int(np.argmax(assignment_fidelity[qubit]))
        frequency[qubit] = data.frequencies[qubit][index[qubit]]

    return ReadoutFrequencyResults(
        assignment_fidelity=assignment_fidelity,
        frequency=frequency,
        angle=angle,
        threshold=threshold,
        index=index,
    )


def _plot(data: ReadoutFrequencyData, fit: ReadoutFrequencyResults, target: QubitId):
    """Plotting function for Optimization RO frequency"""

    figures = []
    fitting_report = ""
    fig = go.Figure()

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=np.array(data.frequencies[target]) * HZ_TO_GHZ,
                y=fit.assignment_fidelity[target],
                mode="markers",
                showlegend=True,
                name="Assignment fidelity",
            )
        )

        fig.add_vline(
            x=data.frequencies[target][fit.index[target]] * HZ_TO_GHZ,
            line_dash="dash",
            name="Best Frequency",
            showlegend=True,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Readout Frequency [GHz]",
                    "Best Assignment Fidelity",
                ],
                [
                    np.round(fit.frequency[target] * HZ_TO_GHZ, 5),
                    np.round(fit.assignment_fidelity[target][fit.index[target]], 3),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Frequency [GHz]",
        yaxis_title="Assignment Fidelity",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ReadoutFrequencyResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_frequency(results.frequency[target], platform, target)
    update.threshold(results.threshold[target][results.index[target]], platform, target)
    update.iq_angle(results.angle[target][results.index[target]], platform, target)


readout_frequency_optimization = Routine(_acquisition, _fit, _plot, _update)
"""Optimization RO frequency Routine object"""
