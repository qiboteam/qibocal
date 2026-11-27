from dataclasses import dataclass

from qibolab import AcquisitionType, PulseSequence

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import readout_frequency

from ...auto.operation import Results
from ..classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    SingleShotClassificationParameters,
)
from .plot import plot_results

__all__ = ["qutrit_classification"]


@dataclass
class QutritClassificationParameters(SingleShotClassificationParameters):
    """SingleShotClassification runcard inputs."""


@dataclass
class QutritClassificationData(SingleShotClassificationData):
    """Qutrit classification data."""


@dataclass
class QutritClassificationResults(Results):
    """Qutrit classification results"""


def _acquisition(
    params: QutritClassificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QutritClassificationData:
    """
    This Protocol prepares the qubits in 0,1 and 2 states and measures their
    respective I, Q values.

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

    data = QutritClassificationData(nshots=params.nshots)

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.INTEGRATION,
    }
    updates = [
        {
            platform.qubits[q].probe: {
                "frequency": readout_frequency(q, platform, state=1)
            }
        }
        for q in targets
    ]

    results = platform.execute(sequences, **options, updates=updates)

    for state, ro_pulses in zip(states, all_ro_pulses):
        for qubit in targets:
            serial = ro_pulses[qubit]
            result = results[serial]
            data.register_qubit(
                ClassificationType,
                (qubit),
                {
                    "i": result[..., 0],
                    "q": result[..., 1],
                    "state": [state] * params.nshots,
                },
            )

    return data


def _fit(data: QutritClassificationData) -> QutritClassificationResults:
    return QutritClassificationResults()


def _plot(
    data: QutritClassificationData,
    target: QubitId,
    fit: QutritClassificationResults,
):
    figures = plot_results(data, target, 3, None)
    fitting_report = ""
    return figures, fitting_report


qutrit_classification = Protocol(_acquisition, _fit, _plot)
"""Qutrit classification Protocol object."""
