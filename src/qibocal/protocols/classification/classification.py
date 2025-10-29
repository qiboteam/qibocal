from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from qibolab import AcquisitionType, PulseSequence
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from qibocal import update
from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    effective_qubit_temperature,
    format_error_single_cell,
    round_report,
    table_dict,
    table_html,
)

ROC_LENGHT = 800
ROC_WIDTH = 800
DEFAULT_CLASSIFIER = "qubit_fit"

__all__ = [
    "single_shot_classification",
    "SingleShotClassificationData",
    "SingleShotClassificationParameters",
]


def compute_snr(zeros: npt.NDArray, ones: npt.NDArray) -> float:
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

    angle: dict[QubitId, float] = field(default_factory=dict)
    threshold: dict[QubitId, float] = field(default_factory=dict)
    snr: dict[QubitId, float] = field(default_factory=dict)
    assignment_fidelity: dict[QubitId, float] = field(default_factory=dict)
    effective_temperature: dict[QubitId, float] = field(default_factory=dict)
    confusion_matrix: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    states: dict[tuple[QubitId, int], list[float]] = field(default_factory=dict)

    def __contains__(self, key):
        return key in self.angle

    @property
    def readout_fidelity(self):
        return {qubit: 2 * fid - 1 for qubit, fid in self.assignment_fidelity.items()}


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
    angle = {}
    threshold = {}
    assignment_fidelity = {}
    snr = {}
    effective_temperature = {}
    confusion_matrix_ = {}
    states = {}
    for qubit in data.qubits:
        nshots = len(data.data[qubit, 0])
        X = np.vstack([data.data[qubit, 0], data.data[qubit, 1]])
        y = np.array([0] * nshots + [1] * nshots)

        lda = LinearDiscriminantAnalysis().fit(X, y)
        w = lda.coef_[0]
        b = lda.intercept_[0]

        angle[qubit] = -np.arctan2(w[1], w[0])
        threshold[qubit] = -b / np.linalg.norm(w)

        pred_y = lda.predict(X)
        snr[qubit] = float(
            compute_snr(zeros=data.data[qubit, 0], ones=data.data[qubit, 1])
        )
        assignment_fidelity[qubit] = np.array(y == pred_y).sum() / 2 / nshots
        effective_temperature[qubit] = effective_qubit_temperature(
            lda.predict(data.data[qubit, 0]),
            qubit_frequency=data.qubit_frequencies[qubit],
            nshots=nshots,
        )
        confusion_matrix_[qubit] = confusion_matrix(
            y, pred_y, normalize="true"
        ).tolist()

        states[qubit, 0] = np.mean(data.data[qubit, 0], axis=0).tolist()
        states[qubit, 1] = np.mean(data.data[qubit, 1], axis=0).tolist()
    return SingleShotClassificationResults(
        angle=angle,
        threshold=threshold,
        assignment_fidelity=assignment_fidelity,
        effective_temperature=effective_temperature,
        snr=snr,
        confusion_matrix=confusion_matrix_,
        states=states,
    )


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    fitting_report = ""
    df0 = {"I": data.data[target, 0].T[0], "Q": data.data[target, 0].T[1]}
    df1 = {"I": data.data[target, 1].T[0], "Q": data.data[target, 1].T[1]}

    fig0 = px.scatter(
        df0,
        x="I",
        y="Q",
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=["blue"],
    )

    for i in range(3):
        fig0.data[i].legendgroup = "State 0"
        if i == 2:
            fig0.data[i].name = "State 0"
            fig0.data[i].showlegend = True
    fig1 = px.scatter(
        df1,
        x="I",
        y="Q",
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=["red"],
    )

    for i in range(3):
        fig1.data[i].legendgroup = "State 1"
        if i == 2:
            fig1.data[i].name = "State 1"
            fig1.data[i].showlegend = True
    if fit is not None:
        min_x = np.min(np.stack([data.data[target, 0].T[0], data.data[target, 1].T[0]]))
        max_x = np.max(np.stack([data.data[target, 0].T[0], data.data[target, 1].T[0]]))
        min_y = np.min(np.stack([data.data[target, 0].T[1], data.data[target, 1].T[1]]))
        max_y = np.max(np.stack([data.data[target, 0].T[1], data.data[target, 1].T[1]]))
        xrange = np.linspace(min_x, max_x, 10000)
        y = (-fit.threshold[target] + xrange * np.cos(fit.angle[target])) / np.sin(
            fit.angle[target]
        )
        indices = np.where(np.logical_and(y > min_y, y < max_y))
        fig0.add_traces(fig1.data)
        fig0.add_trace(
            go.Scatter(
                x=xrange[indices],
                y=y[indices],
                name="Separation",
                line=dict(color="black", dash="dot"),
            )
        )
        # TODO: Plot mean point
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Rotational Angle",
                    "Threshold",
                    "Assignment Fidelity",
                    "Readout Fidelity",
                    "SNR",
                    "Effective Temperature [K]",
                ],
                [
                    np.round(fit.angle[target], 3),
                    np.round(fit.threshold[target], 3),
                    np.round(fit.assignment_fidelity[target], 3),
                    np.round(fit.readout_fidelity[target], 3),
                    np.round(fit.snr[target], 1),
                    format_error_single_cell(
                        round_report([fit.effective_temperature[target]])
                    ),
                ],
            )
        )
    matrix = px.imshow(
        fit.confusion_matrix[target],
        x=["Positive", "Negative"],
        y=["Positive", "Negative"],
        aspect="auto",
        text_auto=True,
        color_continuous_scale="Mint",
        title="Confusion matrix",
    )
    return [fig0, matrix], fitting_report


def _update(
    results: SingleShotClassificationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.iq_angle(results.angle[target], platform, target)
    update.threshold(results.threshold[target], platform, target)
    update.mean_gnd_states(results.states[target, 0], platform, target)
    update.mean_exc_states(results.states[target, 1], platform, target)
    update.readout_fidelity(results.readout_fidelity[target], platform, target)
    platform.calibration.single_qubits[
        target
    ].readout.effective_temperature = results.effective_temperature[target][0]


single_shot_classification = Routine(_acquisition, _fit, _plot, _update)
"""Qubit classification routine object."""
