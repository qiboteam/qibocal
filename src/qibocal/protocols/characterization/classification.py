import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic.dataclasses import Field
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from sklearn.metrics import roc_auc_score, roc_curve

from qibocal.auto.operation import (
    RESULTSFILE,
    Data,
    Parameters,
    Qubits,
    Results,
    Routine,
)
from qibocal.fitting.classifier import run
from qibocal.protocols.characterization.utils import get_color_state0, get_color_state1

MESH_SIZE = 50
MARGIN = 0
COLUMNWIDTH = 600
ROC_LENGHT = 800
ROC_WIDTH = 800
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    classifiers_list: Optional[list[str]] = field(default_factory=lambda: ["qubit_fit"])
    """List of models to classify the qubit states"""
    savedir: Optional[str] = " "
    """Dumping folder of the classification results"""


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64), ("state", int)])
"""Custom dtype for rabi amplitude."""


class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    classifiers_list: Optional[list[str]] = Field(default_factory=lambda: ["qubit_fit"])
    """List of models to classify the qubit states"""
    hpars: dict[QubitId, dict]
    """Models' hyperparameters"""
    savedir: str
    """Dumping folder of the classification results"""
    data: dict[QubitId, npt.NDArray] = Field(default_factory=dict)
    """Raw data acquired."""
    dtype: np.dtype = ClassificationType

    def register_qubit(self, qubit, state, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=ClassificationType)
        ar["i"] = i
        ar["q"] = q
        ar["state"] = state
        self.data[qubit] = np.rec.array(ar)

    def add_data(self, qubit, state, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=ClassificationType)
        ar["i"] = i
        ar["q"] = q
        ar["state"] = state
        self.data[qubit] = np.append(self.data[qubit], np.rec.array(ar))


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""

    y_tests: dict[QubitId, list]
    """States of the testing set."""
    x_tests: dict[QubitId, list]
    """I,Q couples to evaluate accuracy and test time."""
    names: list
    """List of models name."""
    threshold: dict[QubitId, float] = field(metadata=dict(update="threshold"))
    """Threshold for classification."""
    rotation_angle: dict[QubitId, float] = field(metadata=dict(update="iq_angle"))
    """Threshold for classification."""
    mean_gnd_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_gnd_states")
    )
    """Centroid of the ground state blob."""
    mean_exc_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_exc_states")
    )
    """Centroid of the excited state blob."""
    fidelity: dict[QubitId, float]
    """Fidelity evaluated only with the `qubit_fit` model."""
    assignment_fidelity: dict[QubitId, float]
    """Assignment fidelity evaluated only with the `qubit_fit` model."""
    savedir: str
    """Dumping folder of the classification results."""
    y_preds: dict[QubitId, list]
    """Models' predictions of the test set."""
    grid_preds: dict[QubitId, list]
    """Models' prediction of the contour grid."""
    models: dict[QubitId, list] = field(default_factory=list)
    """List of trained classification models."""
    benchmark_table: Optional[dict[QubitId, pd.DataFrame]] = field(default_factory=dict)
    """Benchmark tables."""
    hpars: Optional[dict[QubitId, dict]] = field(
        metadata=dict(update="classifiers_hpars"), default_factory=dict
    )
    """Classifiers hyperparameters."""

    def save(self, path):
        classifiers = run.import_classifiers(self.names)
        for qubit in self.models:
            for i, mod in enumerate(classifiers):
                if self.savedir == " ":
                    save_path = pathlib.Path(path)
                else:
                    save_path = pathlib.Path(self.savedir)

                classifier = run.Classifier(mod, save_path / f"qubit{qubit}")
                classifier.savedir.mkdir(parents=True, exist_ok=True)
                dump_dir = classifier.base_dir / classifier.name / classifier.name
                classifier.dump()(self.models[qubit][i], dump_dir)
                classifier.dump_hyper(self.hpars[qubit][classifier.name])
        asdict_class = asdict(self)
        asdict_class.pop("models")
        asdict_class.pop("hpars")
        (path / RESULTSFILE).write_text(json.dumps(asdict_class, indent=4))


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: Platform,
    qubits: Qubits,
) -> SingleShotClassificationData:
    """
    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers, the available ones are:
            - linear_svm
            - ada_boost
            - gaussian_process
            - naive_bayes
            - nn
            - qubit_fit
            - random_forest
            - rbf_svm
            - qblox_fit.
        The default value is `["qubit_fit"]`.
        savedir (str): Dumping folder of the classification results.
        If not given the dumping folder will be the report one.
        relaxation_time (float): Relaxation time.

        Example:
        .. code-block:: yaml

            - id: single_shot_classification_1
                priority: 0
                operation: single_shot_classification
                parameters:
                nshots: 5000
                savedir: "single_shot"
                classifiers_list: ["qubit_fit","naive_bayes", "linear_svm"]

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    hpars = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])
        hpars[qubit] = qubits[qubit].classifiers_hpars
    # create a DataUnits object to store the results
    data = SingleShotClassificationData(
        nshots=params.nshots,
        classifiers_list=params.classifiers_list,
        hpars=hpars,
        savedir=params.savedir,
    )

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state0_results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit, state=0, i=result.voltage_i, q=result.voltage_q
        )

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )
    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state1_results[ro_pulses[qubit].serial]
        data.add_data(qubit=qubit, state=1, i=result.voltage_i, q=result.voltage_q)

    return data


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits

    benchmark_tables = {}
    models_dict = {}
    y_tests = {}
    x_tests = {}
    hpars = {}
    threshold = {}
    rotation_angle = {}
    mean_gnd_states = {}
    mean_exc_states = {}
    fidelity = {}
    assignment_fidelity = {}
    y_test_predict = {}
    grid_preds_dict = {}
    for qubit in qubits:
        qubit_data = data.data[qubit]
        benchmark_table, y_test, x_test, models, names, hpars_list = run.train_qubit(
            data, qubit
        )
        benchmark_tables[qubit] = benchmark_table.values.tolist()
        models_dict[qubit] = models
        y_tests[qubit] = y_test.tolist()
        x_tests[qubit] = x_test.tolist()
        hpars[qubit] = {}
        y_preds = []
        grid_preds = []
        state0_data = qubit_data[qubit_data["state"] == 0]
        state1_data = qubit_data[qubit_data["state"] == 1]

        grid, q_shape = evaluate_grid(state0_data, state1_data)
        for i, model_name in enumerate(names):
            hpars[qubit][model_name] = hpars_list[i]
            try:
                y_preds.append(models[i].predict_proba(x_test)[:, 1].tolist())
            except AttributeError:
                y_preds.append(models[i].predict(x_test).tolist())
            grid_preds.append(
                np.round(np.reshape(models[i].predict(grid), q_shape))
                .astype(np.int64)
                .tolist()
            )
            if model_name == "qubit_fit":
                threshold[qubit] = models[i].threshold
                rotation_angle[qubit] = models[i].angle
                mean_gnd_states[qubit] = models[i].iq_mean0.tolist()
                mean_exc_states[qubit] = models[i].iq_mean1.tolist()
                fidelity[qubit] = models[i].fidelity
                assignment_fidelity[qubit] = models[i].assignment_fidelity
        y_test_predict[qubit] = y_preds
        grid_preds_dict[qubit] = grid_preds

    return SingleShotClassificationResults(
        benchmark_table=benchmark_tables,
        y_tests=y_tests,
        x_tests=x_tests,
        names=names,
        hpars=hpars,
        models=models_dict,
        threshold=threshold,
        rotation_angle=rotation_angle,
        mean_gnd_states=mean_gnd_states,
        mean_exc_states=mean_exc_states,
        fidelity=fidelity,
        assignment_fidelity=assignment_fidelity,
        savedir=data.savedir,
        y_preds=y_test_predict,
        grid_preds=grid_preds_dict,
    )


def _plot(
    data: SingleShotClassificationData, qubit, fit: SingleShotClassificationResults
):
    figures = []
    fitting_report = ""
    models_name = data.classifiers_list
    qubit_data = data.data[qubit]
    state0_data = qubit_data[qubit_data["state"] == 0]
    state1_data = qubit_data[qubit_data["state"] == 1]
    grid, _ = evaluate_grid(state0_data, state1_data)

    fig = make_subplots(
        rows=1,
        cols=len(models_name),
        horizontal_spacing=SPACING * 3 / len(models_name),
        vertical_spacing=SPACING,
        subplot_titles=[run.pretty_name(model) for model in models_name],
        column_width=[COLUMNWIDTH] * len(models_name),
    )
    fig_roc = go.Figure()
    fig_roc.add_shape(
        type="line", line=dict(dash="dash"), x0=0.0, x1=1.0, y0=0.0, y1=1.0
    )

    if len(models_name) != 1:
        fig_benchmarks = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=SPACING,
            vertical_spacing=SPACING,
            subplot_titles=("accuracy", "training time (s)", "testing time (s)"),
            # pylint: disable=E1101
        )

    if fit is not None:
        y_test = fit.y_tests[qubit]
        x_test = fit.x_tests[qubit]

    for i, model in enumerate(models_name):
        if fit is not None:
            y_pred = fit.y_preds[qubit][i]
            predictions = fit.grid_preds[qubit][i]
            # Evaluate the ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)
            name = f"{model} (AUC={auc_score:.2f})"
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=name,
                    mode="lines",
                    marker=dict(size=3, color=get_color_state0(i)),
                )
            )
            fig.add_trace(
                go.Contour(
                    x=grid[:, 0],
                    y=grid[:, 1],
                    z=np.array(predictions).flatten(),
                    showscale=False,
                    colorscale=[get_color_state0(i), get_color_state1(i)],
                    opacity=0.2,
                    name="Score",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=i + 1,
            )

        model = run.pretty_name(model)
        max_x = max(grid[:, 0])
        max_y = max(grid[:, 1])
        min_x = min(grid[:, 0])
        min_y = min(grid[:, 1])

        fig.add_trace(
            go.Scatter(
                x=state0_data["i"],
                y=state0_data["q"],
                name=f"{model}: state 0",
                legendgroup=f"{model}: state 0",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state0(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=state1_data["i"],
                y=state1_data["q"],
                name=f"{model}: state 1",
                legendgroup=f"{model}: state 1",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state1(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[np.average(state0_data["i"])],
                y=[np.average(state0_data["q"])],
                name=f"{model}: state 0",
                legendgroup=f"{model}: state 0",
                showlegend=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state0(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[np.average(state1_data["i"])],
                y=[np.average(state1_data["q"])],
                name=f"{model}: state 1",
                legendgroup=f"{model}: state 1",
                showlegend=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state1(i)),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            title_text=f"i (V)",
            range=[min_x, max_x],
            row=1,
            col=i + 1,
            autorange=False,
            rangeslider=dict(visible=False),
        )
        fig.update_yaxes(
            title_text="q (V)",
            range=[min_y, max_y],
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=i + 1,
        )

        if fit is not None:
            if len(models_name) != 1:
                fig_benchmarks.add_trace(
                    go.Scatter(
                        x=[model],
                        y=[fit.benchmark_table[qubit][i][0]],
                        mode="markers",
                        showlegend=False,
                        marker=dict(size=10, color=get_color_state1(i)),
                    ),
                    row=1,
                    col=1,
                )

            fig_benchmarks.add_trace(
                go.Scatter(
                    x=[model],
                    y=[fit.benchmark_table[qubit][i][2]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=get_color_state1(i)),
                ),
                row=1,
                col=2,
            )

            fig_benchmarks.add_trace(
                go.Scatter(
                    x=[model],
                    y=[fit.benchmark_table[qubit][i][1]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=get_color_state1(i)),
                ),
                row=1,
                col=3,
            )

            fig_benchmarks.update_yaxes(type="log", row=1, col=2)
            fig_benchmarks.update_yaxes(type="log", row=1, col=3)
            fig_benchmarks.update_layout(
                autosize=False,
                height=COLUMNWIDTH,
                width=COLUMNWIDTH * 3,
                title=dict(text="Benchmarks", font=dict(size=TITLE_SIZE)),
            )

            if models_name[i] == "qubit_fit":
                fitting_report += f"{qubit} | average state 0: {np.round(fit.mean_gnd_states[qubit], 3)}<br>"
                fitting_report += f"{qubit} | average state 1: {np.round(fit.mean_exc_states[qubit], 3)}<br>"
                fitting_report += (
                    f"{qubit} | rotation angle: {fit.rotation_angle[qubit]:.3f}<br>"
                )
                fitting_report += f"{qubit} | threshold: {fit.threshold[qubit]:.6f}<br>"
                fitting_report += f"{qubit} | fidelity: {fit.fidelity[qubit]:.3f}<br>"
                fitting_report += f"{qubit} | assignment fidelity: {fit.assignment_fidelity[qubit]:.3f}<br>"

        fig.update_layout(
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            autosize=False,
            height=COLUMNWIDTH,
            width=COLUMNWIDTH * len(models_name),
            title=dict(text="Results", font=dict(size=TITLE_SIZE)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="left",
                y=-0.3,
                x=0,
                itemsizing="constant",
                font=dict(size=LEGEND_FONT_SIZE),
            ),
        )

        if fit is not None:
            fig_roc.update_layout(
                width=ROC_WIDTH,
                height=ROC_LENGHT,
                title=dict(text="ROC curves", font=dict(size=TITLE_SIZE)),
                legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
            )
            fig_roc.update_xaxes(
                title_text=f"False Positive Rate",
                range=[0, 1],
            )
            fig_roc.update_yaxes(
                title_text="True Positive Rate",
                range=[0, 1],
            )
    figures.append(fig)
    if fit is not None:
        figures.append(fig_roc)
        if len(models_name) != 1:
            figures.append(fig_benchmarks)
    return figures, fitting_report


single_shot_classification = Routine(_acquisition, _fit, _plot)


def evaluate_grid(state0_data, state1_data):
    max_x = (
        max(
            0,
            state0_data["i"].max(),
            state1_data["i"].max(),
        )
        + MARGIN
    )
    max_y = (
        max(
            0,
            state0_data["q"].max(),
            state1_data["q"].max(),
        )
        + MARGIN
    )
    min_x = (
        min(
            0,
            state0_data["i"].min(),
            state1_data["i"].min(),
        )
        - MARGIN
    )
    min_y = (
        min(
            0,
            state0_data["q"].min(),
            state1_data["q"].min(),
        )
        - MARGIN
    )
    i_values, q_values = np.meshgrid(
        np.linspace(min_x, max_x, num=MESH_SIZE),
        np.linspace(min_y, max_y, num=MESH_SIZE),
    )
    return np.vstack([i_values.ravel(), q_values.ravel()]).T, q_values.shape
