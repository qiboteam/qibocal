from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, Parameter, PulseSequence, Readout, Sweeper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.readout.readout_characterization import readout_sequence

# from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import (
    classify,
    compute_assignment_fidelity,
    compute_qnd,
    table_dict,
    table_html,
)

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
    delay: float = 1000
    """Delay between readouts, could account for resonator depletion or not [ns]."""

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
    data: dict[tuple[QubitId, int, int], np.ndarray] = field(default_factory=dict)


@dataclass
class ReadoutAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    assignment_fidelity: dict[QubitId, list]
    """Assignment fidelities."""
    qnd: dict[QubitId, list]
    """QND."""
    qnd_pi: dict[QubitId, list]
    """QND-Pi."""
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
    data = ReadoutAmplitudeData(
        nshots=params.nshots,
        amplitude=params.amplitude_range,
    )

    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += readout_sequence(
                platform=platform,
                delay=params.delay,
                qubit=qubit,
                state=state,
            )

        sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_min,
                params.amplitude_max,
                params.amplitude_step,
            ),
            pulses=[readout[1] for readout in sequence.acquisitions],
        )

        results = platform.execute(
            [sequence],
            [[sweeper]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )

        for target in targets:
            readouts = [
                pulse
                for pulse in sequence.channel(platform.qubits[target].acquisition)
                if isinstance(pulse, Readout)
            ]
            for j, ro_pulse in enumerate(readouts):
                data.data[target, state, j] = results[ro_pulse.id]

    return data


def _fit(data: ReadoutAmplitudeData) -> ReadoutAmplitudeResults:
    angle = {qubit: [] for qubit in data.qubits}
    threshold = {qubit: [] for qubit in data.qubits}
    assignment_fidelity = {qubit: [] for qubit in data.qubits}
    qnd = {qubit: [] for qubit in data.qubits}
    qnd_pi = {qubit: [] for qubit in data.qubits}
    amplitude = {}
    nshots = data.nshots
    y = np.array([0] * nshots + [1] * nshots)
    index = {}
    for qubit in data.qubits:
        for i in range(len(data.amplitude)):
            X = np.vstack(
                [data.data[qubit, 0, 0][:, i, :], data.data[qubit, 1, 0][:, i, :]]
            )
            lda = LinearDiscriminantAnalysis().fit(X, y)
            w = lda.coef_[0]
            b = lda.intercept_[0]

            angle[qubit].append(-np.arctan2(w[1], w[0]))
            threshold[qubit].append(-b / np.linalg.norm(w))

            shots = {0: [], 1: []}
            for state in range(2):
                for m in range(3):
                    shots[state].append(
                        classify(
                            data.data[qubit, state, m][:, i, :],
                            angle[qubit][i],
                            threshold[qubit][i],
                        )
                    )

            assignment_fidelity[qubit].append(
                compute_assignment_fidelity(shots[1][0], shots[0][0])
            )
            qnd_, qnd_pi_ = compute_qnd(shots[0], shots[1])
            qnd[qubit].append(qnd_)
            qnd_pi[qubit].append(qnd_pi_)

        index[qubit] = int(
            np.argmax(
                [
                    (qnd[qubit][i] + qnd_pi[qubit][i]) / 2
                    for i in range(len(data.amplitude))
                ]
            )
        )
        amplitude[qubit] = data.amplitude[index[qubit]]

    return ReadoutAmplitudeResults(
        assignment_fidelity=assignment_fidelity,
        qnd=qnd,
        qnd_pi=qnd_pi,
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
                showlegend=True,
                name="Assignment fidelity",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(data.amplitude),
                y=fit.qnd[target],
                showlegend=True,
                name="QND",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(data.amplitude),
                y=fit.qnd_pi[target],
                showlegend=True,
                name="QND-Pi",
            ),
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
                    "Best QND",
                    "Best QND Pi",
                ],
                [
                    np.round(fit.amplitude[target], 5),
                    np.round(fit.assignment_fidelity[target][fit.index[target]], 3),
                    np.round(fit.qnd[target][fit.index[target]], 3),
                    np.round(fit.qnd_pi[target][fit.index[target]], 3),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude",
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
