from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, Delay, Parameter, PulseSequence, Sweeper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

# from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html

__all__ = ["readout_amplitude_optimization"]


@dataclass
class ReadoutAmplitudeParameters(Parameters):
    """ReadoutAmplitude runcard inputs."""

    amplitude_step: float
    """Amplituude step to be probed."""
    amplitude_min: float = 0.0
    """Amplitude start."""
    amplitude_max: float = 1.0
    """Amplitude stop value"""

    @property
    def amplitude_range(self) -> list[float]:
        return np.arange(
            self.amplitude_min, self.amplitude_max, self.amplitude_step
        ).tolist()


@dataclass
class ReadoutAmplitudeData(Data):
    """Data class for `resoantor_amplitude` protocol."""

    nshots: int
    amplitude: list[float]
    data: dict[tuple[QubitId, int], np.ndarray] = field(default_factory=dict)


@dataclass
class ReadoutAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    assignment_fidelity: dict[QubitId, list]
    """Assignment fidelities."""
    angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""
    amplitude: dict[QubitId, float]
    """Best amplitude."""
    index: dict[QubitId, int]
    """Index corresponding to best fidelity."""


def _acquisition(
    params: ReadoutAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ReadoutAmplitudeParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ReadoutAmplitudeData`)
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

    data = ReadoutAmplitudeData(
        nshots=params.nshots,
        amplitude=params.amplitude_range,
    )

    sweeper_0 = Sweeper(
        parameter=Parameter.amplitude,
        range=(
            params.amplitude_min,
            params.amplitude_max,
            params.amplitude_step,
        ),
        pulses=[ro[1] for ro in sequence_0.acquisitions],
    )

    sweeper_1 = Sweeper(
        parameter=Parameter.amplitude,
        range=(
            params.amplitude_min,
            params.amplitude_max,
            params.amplitude_step,
        ),
        pulses=[ro[1] for ro in sequence_1.acquisitions],
    )

    state0_results = platform.execute(
        [sequence_0],
        [[sweeper_0]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    state1_results = platform.execute(
        [sequence_1],
        [[sweeper_1]],
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


def _fit(data: ReadoutAmplitudeData) -> ReadoutAmplitudeResults:
    angle = {qubit: [] for qubit in data.qubits}
    threshold = {qubit: [] for qubit in data.qubits}
    assignment_fidelity = {qubit: [] for qubit in data.qubits}
    amplitude = {}
    nshots = data.nshots
    y = np.array([0] * nshots + [1] * nshots)
    index = {}
    for qubit in data.qubits:
        for i in range(len(data.amplitude)):
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
        amplitude[qubit] = data.amplitude[index[qubit]]

    return ReadoutAmplitudeResults(
        assignment_fidelity=assignment_fidelity,
        amplitude=amplitude,
        angle=angle,
        threshold=threshold,
        index=index,
    )


def _plot(data: ReadoutAmplitudeData, fit: ReadoutAmplitudeResults, target: QubitId):
    """Plotting function for Optimization RO amplitude."""
    figures = []
    fitting_report = ""
    fig = go.Figure()

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=np.array(data.amplitude),
                y=fit.assignment_fidelity[target],
                mode="markers",
                showlegend=True,
                name="Assignment fidelity",
            )
        )

        fig.add_vline(
            x=data.amplitude[fit.index[target]],
            line_dash="dash",
            name="Best Amplitude",
            showlegend=True,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Readout Amplitude",
                    "Best Assignment Fidelity",
                ],
                [
                    np.round(fit.amplitude[target], 5),
                    np.round(fit.assignment_fidelity[target][fit.index[target]], 3),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude",
        yaxis_title="Assignment Fidelity",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ReadoutAmplitudeResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_amplitude(results.amplitude[target], platform, target)
    update.threshold(results.threshold[target][results.index[target]], platform, target)
    update.iq_angle(results.angle[target][results.index[target]], platform, target)


readout_amplitude_optimization = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Amplitude Routine  object."""
