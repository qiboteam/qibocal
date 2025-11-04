from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    Parameter,
    PulseSequence,
    Readout,
    Sweeper,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    classify,
    compute_assignment_fidelity,
    compute_qnd,
    readout_frequency,
    table_dict,
    table_html,
)

from ..readout.readout_characterization import readout_sequence

__all__ = ["readout_frequency_optimization"]


@dataclass
class ReadoutFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    delay: int = 1000
    """Delay after measurement [ns]."""

    @property
    def frequency_range(self) -> np.ndarray:
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)


@dataclass
class ReadoutFrequencyResults(Results):
    """Optimization RO frequency results."""

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
    frequency: dict[QubitId, float]
    """Best frequency."""
    index: dict[QubitId, int]
    """Index corresponding to best fidelity."""


@dataclass
class ReadoutFrequencyData(Data):
    """Optimization RO frequency acquisition outputs."""

    nshots: int
    frequencies: dict[QubitId, list[float]] = field(default_factory=dict)
    data: dict[tuple[QubitId, int, int], np.ndarray] = field(default_factory=dict)


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
    data = ReadoutFrequencyData(
        nshots=params.nshots,
        frequencies={
            q: (readout_frequency(q, platform) + params.frequency_range).tolist()
            for q in targets
        },
    )

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + params.frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    for state in range(2):
        sequence = PulseSequence()
        for qubit in targets:
            sequence += readout_sequence(
                platform=platform,
                delay=params.delay,
                qubit=qubit,
                state=state,
            )

        results = platform.execute(
            [sequence],
            [sweepers],
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


def _fit(data: ReadoutFrequencyData) -> ReadoutFrequencyResults:
    """Post-Processing for Optimization RO frequency"""

    angle = {qubit: [] for qubit in data.qubits}
    threshold = {qubit: [] for qubit in data.qubits}
    assignment_fidelity = {qubit: [] for qubit in data.qubits}
    qnd = {qubit: [] for qubit in data.qubits}
    qnd_pi = {qubit: [] for qubit in data.qubits}
    frequency = {}
    nshots = data.nshots
    y = np.array([0] * nshots + [1] * nshots)
    index = {}
    for qubit in data.qubits:
        for i in range(len(data.frequencies[qubit])):
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
                    for i in range(len(data.frequencies[qubit]))
                ]
            )
        )
        frequency[qubit] = data.frequencies[qubit][index[qubit]]

    return ReadoutFrequencyResults(
        assignment_fidelity=assignment_fidelity,
        qnd=qnd,
        qnd_pi=qnd_pi,
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
                showlegend=True,
                name="Assignment fidelity",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(data.frequencies[target]) * HZ_TO_GHZ,
                y=fit.qnd[target],
                showlegend=True,
                name="QND",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(data.frequencies[target]) * HZ_TO_GHZ,
                y=fit.qnd_pi[target],
                showlegend=True,
                name="QND-Pi",
            ),
        )

        fig.add_vline(
            x=data.frequencies[target][fit.index[target]] * HZ_TO_GHZ,
            line_dash="dash",
            name="Best Readout Frequency",
            showlegend=True,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Readout Frequency [GHz]",
                    "Best Assignment Fidelity",
                    "Best QND",
                    "Best QND Pi",
                ],
                [
                    np.round(fit.frequency[target] * HZ_TO_GHZ, 5),
                    np.round(fit.assignment_fidelity[target][fit.index[target]], 3),
                    np.round(fit.qnd[target][fit.index[target]], 3),
                    np.round(fit.qnd_pi[target][fit.index[target]], 3),
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
