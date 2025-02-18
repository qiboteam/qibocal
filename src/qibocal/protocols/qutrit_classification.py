from dataclasses import dataclass, field
from typing import Optional

from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Results, Routine
from qibocal.protocols.classification import (
    ClassificationType,
    SingleShotClassificationData,
    SingleShotClassificationParameters,
)
from qibocal.protocols.utils import plot_results

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
    states_sequences = [PulseSequence() for _ in range(3)]
    ro_pulses = {}
    for qubit in targets:
        rx_pulse = platform.create_RX_pulse(qubit, start=0)
        rx12_pulse = platform.create_RX12_pulse(qubit, start=rx_pulse.finish)
        drive_pulses = [rx_pulse, rx12_pulse]
        ro_pulses[qubit] = []
        for i, sequence in enumerate(states_sequences):
            sequence.add(*drive_pulses[:i])
            start = drive_pulses[i - 1].finish if i != 0 else 0
            ro_pulses[qubit].append(
                platform.create_qubit_readout_pulse(qubit, start=start)
            )
            sequence.add(ro_pulses[qubit][-1])

    data = QutritClassificationData(
        nshots=params.nshots,
        classifiers_list=params.classifiers_list,
        savedir=params.savedir,
    )
    states_results = []
    for sequence in states_sequences:
        states_results.append(
            platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )
        )

    for qubit in targets:
        for state, state_result in enumerate(states_results):
            result = state_result[ro_pulses[qubit][state].serial]
            data.register_qubit(
                ClassificationType,
                (qubit),
                dict(
                    state=[state] * params.nshots,
                    i=result.voltage_i,
                    q=result.voltage_q,
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
