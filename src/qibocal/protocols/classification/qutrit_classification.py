from dataclasses import dataclass

from qibolab import AcquisitionType, PulseSequence

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import readout_frequency

from ...auto.operation import Results
from ..classification.classification import (
    SingleShotClassificationData,
    SingleShotClassificationParameters,
)
from .utils import plot

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
    return QutritClassificationResults()


def _plot(
    data: QutritClassificationData,
    target: QubitId,
    fit: QutritClassificationResults,
):
    fig0 = plot(
        data={"I": data.data[target, 0].T[0], "Q": data.data[target, 0].T[1]},
        color="red",
        label="State 0",
    )

    fig1 = plot(
        data={"I": data.data[target, 1].T[0], "Q": data.data[target, 1].T[1]},
        color="blue",
        label="State 1",
    )

    fig2 = plot(
        data={"I": data.data[target, 2].T[0], "Q": data.data[target, 2].T[1]},
        color="green",
        label="State 2",
    )

    fig0.add_traces(fig1.data)
    fig0.add_traces(fig2.data)
    return [fig0], ""


qutrit_classification = Routine(_acquisition, _fit, _plot)
"""Qutrit classification Routine object."""
