from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, PulseSequence
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

classifiers = {
    "knn": KNeighborsClassifier(5),
    "linear_svc": SVC(kernel="linear", C=0.025, random_state=42),
    "rbf_svc": SVC(gamma=2, C=1, random_state=42),
    "gaussian_process": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    "neural_net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "adaboost": AdaBoostClassifier(random_state=42),
    "naive_bayes": GaussianNB(),
    "lda": LinearDiscriminantAnalysis(),
    "qda": QuadraticDiscriminantAnalysis(),
}

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

from .utils import plot_confusion_matrix, plot_distribution
from matplotlib.figure import Figure

ROC_LENGHT = 800
ROC_WIDTH = 800
DEFAULT_CLASSIFIER = "lda"

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

    classifier: str = DEFAULT_CLASSIFIER
    """Classifier to use. Available options are:
    - "knn"
    - "linear_svc"
    - "rbf_svc"
    - "gaussian_process"
    - "decision_tree"
    - "random_forest"
    - "neural_net"
    - "adaboost"
    - "naive_bayes"
    - "lda"
    - "qda" 
    """


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    classifier: str = DEFAULT_CLASSIFIER
    """Classifier used."""
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
    classifier: str = DEFAULT_CLASSIFIER

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
        classifier=params.classifier,
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

        clf = classifiers[data.classifier].fit(X, y)
        w = clf.coef_[0] if hasattr(clf, "coef_") else np.zeros(2)
        b = clf.intercept_[0] if hasattr(clf, "intercept_") else 0.0

        angle[qubit] = -np.arctan2(w[1], w[0])
        threshold[qubit] = -b / np.linalg.norm(w)

        pred_y = clf.predict(X)
        snr[qubit] = float(
            compute_snr(zeros=data.data[qubit, 0], ones=data.data[qubit, 1])
        )
        assignment_fidelity[qubit] = np.array(y == pred_y).sum() / 2 / nshots
        effective_temperature[qubit] = effective_qubit_temperature(
            clf.predict(data.data[qubit, 0]),
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
        classifier = data.classifier,
    )


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    fitting_report = ""
    figures = []
    colors = ["red", "blue"]
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )
    nshots = len(data.data[target, 0])
    X = np.vstack([data.data[target, 0], data.data[target, 1]])
    y = np.array([0] * nshots + [1] * nshots)
    clf = classifiers[data.classifier].fit(X, y) # Train this again to plot he boundary

    for state in [0, 1]:
        plot_distribution(
            fig=fig,
            data={
                "I": data.data[target, state].T[0],
                "Q": data.data[target, state].T[1],
            },
            color=colors[state],
            label=f"State {state}",
        )

    fig.update_layout(
        hovermode="closest",
        barmode="overlay",
        xaxis3_title="I",
        yaxis3_title="Q",
    )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=2, col=1)

    # fig0.add_traces(fig1.data)
    figures.append(fig)
    if fit is not None:
        min_x = np.min(np.stack([data.data[target, 0].T[0], data.data[target, 1].T[0]]))
        max_x = np.max(np.stack([data.data[target, 0].T[0], data.data[target, 1].T[0]]))
        min_y = np.min(np.stack([data.data[target, 0].T[1], data.data[target, 1].T[1]]))
        max_y = np.max(np.stack([data.data[target, 0].T[1], data.data[target, 1].T[1]]))
        
        if fit.threshold[target] is not None and fit.angle[target] is not None:
            x = np.linspace(min_x, max_x, 100)
            y = (-fit.threshold[target] + x * np.cos(fit.angle[target])) / np.sin(
                fit.angle[target]
            )
            indices = np.where(np.logical_and(y > min_y, y < max_y))
            fig.add_trace(
                go.Scatter(
                    x=x[indices],
                    y=y[indices],
                    name="Separation",
                    line=dict(color="black", dash="dot"),
                ),
                row=2,
                col=1,
            )
       
        # Plot decision boundary manually (Decision boundary from sklearn uses matplotlib)
        X1, X2 = np.meshgrid(
            np.linspace(min_x, max_x, 200), np.linspace(min_y, max_y, 200)
        )
        Z = clf.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

        fig.add_trace(
            go.Contour(
            x=np.linspace(min_x, max_x, 200),
            y=np.linspace(min_y, max_y, 200),
            z=Z,
            showscale=False,
            colorscale=[colors[0], colors[1]],
            hoverinfo="skip",
            line_width=0,
            opacity=0.3,
            ),
            row=2,
            col=1,
        )

        # Average states
        for state in [0, 1]:
            fig.add_scatter(
                x=[np.mean(data.data[target, state].T[0])],
                y=[np.mean(data.data[target, state].T[1])],
                mode="markers",
                marker=dict(color=f"dark{colors[state]}", symbol="x", size=12),
                name=f"Average State |{state}>",
                row=2,
                col=1,
            )

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
        figures.append(
            plot_confusion_matrix(
                confusion_matrix=fit.confusion_matrix[target], labels=["0", "1"]
            )
        )
    return figures, fitting_report


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
