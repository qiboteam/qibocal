from dataclasses import dataclass, field
from typing import Optional

from qibolab import AcquisitionType, Platform, PulseSequence

from qibocal.auto.operation import QubitId, Routine
from qibocal.protocols.classification import (
    ClassificationType,
    SingleShotClassificationData,
    SingleShotClassificationParameters,
)
from qibocal.protocols.utils import plot_results

from ..auto.operation import Results
from ..config import log

COLUMNWIDTH = 600
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1
DEFAULT_CLASSIFIER = "naive_bayes"


@dataclass
class QutritClassificationParameters(SingleShotClassificationParameters):
    """SingleShotClassification runcard inputs."""


@dataclass
class QutritClassificationData(SingleShotClassificationData):
    classifiers_list: Optional[list[str]] = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states"""


@dataclass
class QutritClassificationResults(Results):
    """Qutrit classification results"""


def _acquisition(
    params: QutritClassificationParameters,
    platform: Platform,
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

    updates = []
    for qubit in targets:
        channel = platform.qubits[qubit].probe
        try:
            updates.append(
                {
                    channel: {
                        "frequency": platform.calibration.single_qubits[
                            qubit
                        ].readout.qudits_frequency[1]
                    }
                }
            )
        except KeyError:
            log.warning(f"No readout frequency for state 1 for qubit {qubit}.")

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
        classifiers_list=params.classifiers_list,
        savedir=params.savedir,
    )

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    updates = []
    for qubit in targets:
        channel = platform.qubits[qubit].probe
        try:
            # we readout at the readout frequency of |1> for better discrimination
            updates.append(
                {
                    channel: {
                        "frequency": platform.calibration.single_qubits[
                            qubit
                        ].readout.qudits_frequency[1]
                    }
                }
            )
        except KeyError:
            log.warning(f"No readout frequency for state 1 for qubit {qubit}.")

    if params.unrolling:
        results = platform.execute(sequences, **options, updates=updates)
    else:
        results = {}
        for sequence in sequences:
            results.update(platform.execute([sequence], **options, updates=updates))

    for state, ro_pulses in zip(states, all_ro_pulses):
        for qubit in targets:
            serial = ro_pulses[qubit]
            result = results[serial]
            data.register_qubit(
                ClassificationType,
                (qubit),
                dict(
                    i=result[..., 0],
                    q=result[..., 1],
                    state=[state] * params.nshots,
                ),
            )

    return data


def _fit(data: QutritClassificationData) -> QutritClassificationResults:
    return QutritClassificationResults()


def _plot(
    data: QutritClassificationData,
    target: QubitId,
    fit: QutritClassificationResults,
):
    figures = plot_results(data, target, 3, fit)
    fitting_report = ""
    return figures, fitting_report


qutrit_classification = Routine(_acquisition, _fit, _plot)
"""Qutrit classification Routine object."""
