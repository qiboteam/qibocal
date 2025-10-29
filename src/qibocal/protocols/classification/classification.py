from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.express as px
from qibolab import AcquisitionType, PulseSequence

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform

ROC_LENGHT = 800
ROC_WIDTH = 800
DEFAULT_CLASSIFIER = "qubit_fit"

__all__ = [
    "single_shot_classification",
    "SingleShotClassificationData",
    "SingleShotClassificationParameters",
]


def evaluate_snr(zeros: npt.NDArray, ones: npt.NDArray) -> float:
    """Compute snr for zeros and ones"""
    line = np.mean(ones, axis=0) - np.mean(zeros, axis=0)
    projection_zeros, projection_ones = np.dot(zeros, line), np.dot(ones, line)
    mu0, std0 = np.mean(projection_zeros), np.std(projection_zeros)
    mu1, std1 = np.mean(projection_ones), np.std(projection_ones)
    return np.abs(mu1 - mu0) ** 2 / 2 / std0 / std1


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    unrolling: bool = False
    """Whether to unroll the sequences.

    If ``True`` it uses sequence unrolling to deploy multiple sequences in a
    single instrument call.

    Defaults to ``False``.
    """


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    data: dict[tuple[QubitId, int], npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SingleShotClassificationData:
    """
    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers, the available ones are:
            - qubit_fit
            - qblox_fit.
        The default value is `["qubit_fit"]`.
        savedir (str): Dumping folder of the classification results.
        If not given the dumping folder will be the report one.
        relaxation_time (float): Relaxation time.

        Example:
        .. code-block:: yaml

        - id: single_shot_classification_1
            operation: single_shot_classification
            parameters:
            nshots: 5000
            savedir: "single_shot"
            classifiers_list: ["qubit_fit"]

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    native = platform.natives.single_qubit
    sequences, all_ro_pulses = [], []
    for state in [0, 1]:
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

        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    data = SingleShotClassificationData(
        nshots=params.nshots,
        qubit_frequencies={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
    )

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    if params.unrolling:
        results = platform.execute(sequences, **options)
    else:
        results = {}
        for sequence in sequences:
            results.update(platform.execute([sequence], **options))

    for state, ro_pulses in zip([0, 1], all_ro_pulses):
        for qubit in targets:
            data.data[qubit, state] = results[ro_pulses[qubit]]

    return data


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    return SingleShotClassificationResults()


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    fitting_report = ""
    df0 = {"I": data.data[target, 0].T[0], "Q": data.data[target, 0].T[1]}
    df1 = {"I": data.data[target, 1].T[0], "Q": data.data[target, 1].T[1]}
    fig1 = px.scatter(
        df0,
        x="I",
        y="Q",
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=["blue"],
    )
    fig2 = px.scatter(
        df1,
        x="I",
        y="Q",
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=["red"],
    )
    fig1.add_traces(fig2.data)
    return [fig1], fitting_report


def _update(
    results: SingleShotClassificationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    pass
    # update.iq_angle(results.rotation_angle[target], platform, target)
    # update.threshold(results.threshold[target], platform, target)
    # update.mean_gnd_states(results.mean_gnd_states[target], platform, target)
    # update.mean_exc_states(results.mean_exc_states[target], platform, target)
    # update.readout_fidelity(results.fidelity[target], platform, target)
    # platform.calibration.single_qubits[
    #     target
    # ].readout.effective_temperature = results.effective_temperature[target][0]


single_shot_classification = Routine(_acquisition, _fit, _plot, _update)
"""Qubit classification routine object."""
