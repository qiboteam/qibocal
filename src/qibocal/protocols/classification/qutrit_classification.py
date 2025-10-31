from dataclasses import dataclass, field

import numpy as np
from qibolab import AcquisitionType, PulseSequence
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import readout_frequency

from ...auto.operation import Results
from ..classification.classification import (
    SingleShotClassificationData,
    SingleShotClassificationParameters,
)
from .utils import plot_confusion_matrix, plot_distribution

__all__ = ["qutrit_classification"]


@dataclass
class QutritClassificationParameters(SingleShotClassificationParameters):
    """SingleShotClassification runcard inputs."""


@dataclass
class QutritClassificationData(SingleShotClassificationData):
    """Qutrit classification results."""


@dataclass
class QutritClassificationResults(Results):
    """Qutrit classification results"""

    confusion_matrix: dict[QubitId, list[list[float]]] = field(default_factory=dict)


def _acquisition(
    params: QutritClassificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QutritClassificationData:
    """
    This Routine prepares the qubits in 0,1 and 2 states and measures their
    respective I, Q values.

    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers, the available ones are:
            - naive_bayes
            - nn
            - random_forest
            - decision_tree
        The default value is `["naive_bayes"]`.
        savedir (str): Dumping folder of the classification results.
        If not given, the dumping folder will be the report one.
        relaxation_time (float): Relaxation time.
    """

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    states = [0, 1, 2]
    sequences, all_ro_pulses = [], []
    native = platform.natives.single_qubit

    for state in states:
        ro_pulses = {}
        sequence = PulseSequence()
        for q in targets:
            ro_sequence = native[q].MZ()
            ro_pulses[q] = ro_sequence[0][1].id
            sequence += ro_sequence

        if state == 1:
            rx_sequence = PulseSequence()
            for q in targets:
                rx_sequence += native[q].RX()
            sequence = rx_sequence | sequence

        if state == 2:
            rx12_sequence = PulseSequence()
            for q in targets:
                rx12_sequence += native[q].RX() | native[q].RX12()
            sequence = rx12_sequence | sequence

        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    data = QutritClassificationData(
        nshots=params.nshots,
    )

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )
    updates = [
        {
            platform.qubits[q].probe: {
                "frequency": readout_frequency(q, platform, state=1)
            },
        }
        for q in targets
    ]
    if params.unrolling:
        results = platform.execute(sequences, **options, updates=updates)
    else:
        results = {}
        for sequence in sequences:
            results.update(platform.execute([sequence], **options, updates=updates))

    for state, ro_pulses in zip(states, all_ro_pulses):
        for qubit in targets:
            data.data[qubit, state] = results[ro_pulses[qubit]]

    return data


def _fit(data: QutritClassificationData) -> QutritClassificationResults:
    confusion_matrix_ = {}
    for qubit in data.qubits:
        nshots = len(data.data[qubit, 0])
        X = np.vstack([data.data[qubit, i] for i in range(3)])
        y = np.array([0] * nshots + [1] * nshots + [2] * nshots)

        lda = LinearDiscriminantAnalysis().fit(X, y)
        confusion_matrix_[qubit] = confusion_matrix(
            y, lda.predict(X), normalize="true"
        ).tolist()
    return QutritClassificationResults(confusion_matrix=confusion_matrix_)


def _plot(
    data: QutritClassificationData,
    target: QubitId,
    fit: QutritClassificationResults,
):
    figures = []
    fig0 = plot_distribution(
        data={"I": data.data[target, 0].T[0], "Q": data.data[target, 0].T[1]},
        color="red",
        label="State 0",
    )

    fig1 = plot_distribution(
        data={"I": data.data[target, 1].T[0], "Q": data.data[target, 1].T[1]},
        color="blue",
        label="State 1",
    )

    fig2 = plot_distribution(
        data={"I": data.data[target, 2].T[0], "Q": data.data[target, 2].T[1]},
        color="green",
        label="State 2",
    )

    fig0.add_traces(fig1.data)
    fig0.add_traces(fig2.data)
    figures.append(fig0)

    if fit is not None:
        figures.append(
            plot_confusion_matrix(
                confusion_matrix=fit.confusion_matrix[target], labels=["0", "1", "2"]
            )
        )
    return figures, ""


qutrit_classification = Routine(_acquisition, _fit, _plot)
"""Qutrit classification Routine object."""
