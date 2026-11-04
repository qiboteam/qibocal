import json
import pathlib
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from qibolab import AcquisitionType, PulseSequence
from sklearn.metrics import roc_auc_score, roc_curve

from qibocal import update
from qibocal.auto.operation import (
    RESULTSFILE,
    Data,
    Parameters,
    Protocol,
    QubitId,
    Results,
)
from qibocal.auto.serialize import serialize
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.fitting.classifier import run
from qibocal.protocols.utils import (
    LEGEND_FONT_SIZE,
    MESH_SIZE,
    TITLE_SIZE,
    effective_qubit_temperature,
    evaluate_grid,
    format_error_single_cell,
    get_color_state0,
    round_report,
    table_dict,
    table_html,
)

from .mixture import GaussianMixture, fit_mixture
from .outliers import MAX_EXCLUDED, CoreRegion, core_region, rotate
from .plot import plot_results

ROC_LENGHT = 800
ROC_WIDTH = 800
DEFAULT_CLASSIFIER = "qubit_fit"

__all__ = [
    "ClassificationType",
    "SingleShotClassificationData",
    "SingleShotClassificationParameters",
    "single_shot_classification",
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

    classifiers_list: list[str] | None = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states."""
    savedir: str | None = " "
    """Dumping folder of the classification results."""


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64), ("state", int)])
"""Custom dtype for classification."""


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    savedir: str
    """Dumping folder of the classification results."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    classifiers_list: list[str] | None = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states."""

    def state_zero(self, qubit: QubitId) -> npt.NDArray:
        """Get state zero data."""
        state_zero = self.data[qubit][self.data[qubit].state == 0]
        return np.column_stack([state_zero.i, state_zero.q])

    def state_one(self, qubit: QubitId) -> npt.NDArray:
        """Get state one data."""
        state_one = self.data[qubit][self.data[qubit].state == 1]
        return np.column_stack([state_one.i, state_one.q])


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""

    names: list
    """List of models name."""
    savedir: str
    """Dumping folder of the classification results."""
    y_preds: dict[QubitId, list]
    """Models' predictions of the test set."""
    grid_preds: dict[QubitId, list]
    """Models' prediction of the contour grid."""
    threshold: dict[QubitId, float] = field(default_factory=dict)
    """Threshold for classification."""
    rotation_angle: dict[QubitId, float] = field(default_factory=dict)
    """Threshold for classification."""
    mean_gnd_states: dict[QubitId, list[float]] = field(default_factory=dict)
    """Centroid of the ground state blob."""
    mean_exc_states: dict[QubitId, list[float]] = field(default_factory=dict)
    """Centroid of the excited state blob."""
    fidelity: dict[QubitId, float] = field(default_factory=dict)
    """Fidelity evaluated only with the `qubit_fit` model."""
    assignment_fidelity: dict[QubitId, float] = field(default_factory=dict)
    """Assignment fidelity evaluated only with the `qubit_fit` model."""
    effective_temperature: dict[QubitId, float] = field(default_factory=dict)
    """Qubit effective temperature from Boltzmann distribution."""
    models: dict[QubitId, list] = field(default_factory=list)
    """List of trained classification models."""
    benchmark_table: dict[QubitId, pd.DataFrame] | None = field(default_factory=dict)
    """Benchmark tables."""
    classifiers_hpars: dict[QubitId, dict] | None = field(default_factory=dict)
    """Classifiers hyperparameters."""
    x_tests: dict[QubitId, list] = field(default_factory=dict)
    """Test set."""
    y_tests: dict[QubitId, list] = field(default_factory=dict)
    """Test set."""
    snr: dict[QubitId, float] = field(default_factory=dict)
    """SNR for two clouds"""
    confusion: dict[QubitId, list[float]] = field(default_factory=dict)
    """Errors of the `qubit_fit` classifier, P(1|0) and P(0|1), on the prepared
    states."""
    mixture: dict[QubitId, dict] = field(default_factory=dict)
    """Two-Gaussian mixture fit of the projected shots, cf.
    :class:`~qibocal.protocols.classification.mixture.GaussianMixture`."""
    mixture_effective_temperature: dict[QubitId, tuple[float, float]] = field(
        default_factory=dict
    )
    """Qubit effective temperature from the spurious excited population."""
    outliers: dict[QubitId, dict] = field(default_factory=dict)
    """Region of the shots used in the Gaussian mixture fit, cf.
    :class:`~qibocal.protocols.classification.outliers.CoreRegion`."""

    OPTIONAL = (
        "classifiers_hpars",
        "confusion",
        "mixture",
        "mixture_effective_temperature",
        "outliers",
    )
    """Fields possibly missing for some qubits: absent when running only the default
    classifier, in results produced by previous versions, or for failed mixture fits."""

    def __contains__(self, key: QubitId):
        """Checking if key is in Results.

        Overwritten to skip the optional fields.
        """
        return all(
            key in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
            and field.name not in self.OPTIONAL
        )

    def gaussian_mixture(
        self, qubit: QubitId, consistent: bool = False
    ) -> GaussianMixture | None:
        """Two-Gaussian mixture fitted for a qubit, if any.

        With `consistent`, only a fit describing the measured distributions is
        returned.
        """
        parameters = self.mixture.get(qubit)
        if parameters is None:
            return None
        mixture = GaussianMixture(**parameters)
        return mixture if mixture.consistent or not consistent else None

    def core_region(self, qubit: QubitId) -> CoreRegion | None:
        """Region of the shots used in the Gaussian mixture fit, if any."""
        parameters = self.outliers.get(qubit)
        return CoreRegion(**parameters) if parameters is not None else None

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
                classifier.dump_hyper(self.classifiers_hpars[qubit][classifier.name])
        asdict_class = asdict(self)
        asdict_class.pop("models")
        asdict_class.pop("classifiers_hpars")
        (path / f"{RESULTSFILE}.json").write_text(
            json.dumps(serialize(asdict_class)), encoding="utf-8"
        )


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
        classifiers_list=params.classifiers_list,
        savedir=params.savedir,
    )

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.INTEGRATION,
    }

    results = platform.execute(sequences, **options)

    for state, ro_pulses in zip([0, 1], all_ro_pulses):
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


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits

    benchmark_tables = {}
    models_dict = {}
    y_tests = {}
    x_tests = {}
    hpars = {}
    snr = {}
    threshold = {}
    rotation_angle = {}
    mean_gnd_states = {}
    mean_exc_states = {}
    fidelity = {}
    assignment_fidelity = {}
    y_test_predict = {}
    grid_preds_dict = {}
    effective_temperature = {}
    confusion = {}
    mixtures = {}
    mixture_temperature = {}
    regions = {}
    for qubit in qubits:
        qubit_data = data.data[qubit]
        state0_data = qubit_data[qubit_data.state == 0]
        iq_state0 = state0_data[["i", "q"]]
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
        grid = evaluate_grid(qubit_data)
        for i, model_name in enumerate(names):
            hpars[qubit][model_name] = hpars_list[i]
            try:
                y_preds.append(models[i].predict_proba(x_test)[:, 1].tolist())
            except AttributeError:
                y_preds.append(models[i].predict(x_test).tolist())
            grid_preds.append(
                np.round(np.reshape(models[i].predict(grid), (MESH_SIZE, MESH_SIZE)))
                .astype(np.int64)
                .tolist()
            )
            if model_name == "qubit_fit":
                threshold[qubit] = models[i].threshold
                rotation_angle[qubit] = models[i].angle
                mean_gnd_states[qubit] = models[i].iq_mean0
                mean_exc_states[qubit] = models[i].iq_mean1
                fidelity[qubit] = models[i].fidelity
                snr[qubit] = evaluate_snr(data.state_zero(qubit), data.state_one(qubit))
                assignment_fidelity[qubit] = models[i].assignment_fidelity
                predictions_state0 = models[i].predict(iq_state0.tolist())
                effective_temperature[qubit] = models[i].effective_temperature(
                    predictions_state0, data.qubit_frequencies[qubit]
                )
                ground = rotate(data.state_zero(qubit), models[i].angle)
                excited = rotate(data.state_one(qubit), models[i].angle)
                confusion[qubit] = [
                    float(np.mean(ground[:, 0] > models[i].threshold)),
                    float(np.mean(excited[:, 0] <= models[i].threshold)),
                ]
                region = core_region(ground, excited)
                regions[qubit] = region.dump()
                if region.excluded:
                    ground = ground[region.inside(ground)]
                    excited = excited[region.inside(excited)]
                try:
                    mixture = fit_mixture(ground[:, 0], excited[:, 0])
                except ValueError as error:
                    log.warning(
                        f"Gaussian mixture fit failed for qubit {qubit}: {error}"
                    )
                else:
                    mixtures[qubit] = mixture.dump()
                    if not mixture.consistent:
                        log.warning(
                            f"Gaussian mixture inconsistent with the data for qubit "
                            f"{qubit}: not used"
                        )
                    excitation = mixture.spurious[0]
                    mixture_temperature[qubit] = effective_qubit_temperature(
                        prob_0=1 - excitation,
                        prob_1=excitation,
                        qubit_frequency=data.qubit_frequencies[qubit],
                        nshots=len(ground),
                    )
        y_test_predict[qubit] = y_preds
        grid_preds_dict[qubit] = grid_preds
    return SingleShotClassificationResults(
        benchmark_table=benchmark_tables,
        y_tests=y_tests,
        x_tests=x_tests,
        names=names,
        classifiers_hpars=hpars,
        models=models_dict,
        threshold=threshold,
        rotation_angle=rotation_angle,
        mean_gnd_states=mean_gnd_states,
        mean_exc_states=mean_exc_states,
        fidelity=fidelity,
        assignment_fidelity=assignment_fidelity,
        effective_temperature=effective_temperature,
        savedir=data.savedir,
        y_preds=y_test_predict,
        grid_preds=grid_preds_dict,
        snr=snr,
        confusion=confusion,
        mixture=mixtures,
        mixture_effective_temperature=mixture_temperature,
        outliers=regions,
    )


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    fitting_report = ""
    models_name = data.classifiers_list
    figures = plot_results(data, target, 2, fit)
    if fit is not None:
        y_test = fit.y_tests[target]
        y_pred = fit.y_preds[target]

        if len(models_name) != 1:
            # Evaluate the ROC curve
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line", line={"dash": "dash"}, x0=0.0, x1=1.0, y0=0.0, y1=1.0
            )
            for i, model in enumerate(models_name):
                y_pred = fit.y_preds[target][i]
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred)
                name = f"{model} (AUC={auc_score:.2f})"
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        name=name,
                        mode="lines",
                        marker={"size": 3, "color": get_color_state0(i)},
                    )
                )
            fig_roc.update_layout(
                width=ROC_WIDTH,
                height=ROC_LENGHT,
                title={"text": "ROC curves", "font": {"size": TITLE_SIZE}},
                legend={"font": {"size": LEGEND_FONT_SIZE}},
            )
            fig_roc.update_xaxes(
                title_text="False Positive Rate",
                range=[0, 1],
            )
            fig_roc.update_yaxes(
                title_text="True Positive Rate",
                range=[0, 1],
            )
            figures.append(fig_roc)

        if "qubit_fit" in models_name:
            fitting_report = table_html(table_dict(target, *_report_rows(fit, target)))

    return figures, fitting_report


def _report_rows(
    fit: SingleShotClassificationResults, target: QubitId
) -> tuple[list[str], list]:
    """Names and values of the report table.

    With a Gaussian mixture fit, the quantities of the naive linear classifier are
    compared with the ones of the two fitted Gaussians, which exclude the spurious
    populations of each preparation.
    """

    def temperature(value):
        return format_error_single_cell(round_report([value]))

    rows = [
        ("Average State 0", np.round(fit.mean_gnd_states[target], 3)),
        ("Average State 1", np.round(fit.mean_exc_states[target], 3)),
        ("Rotational Angle", np.round(fit.rotation_angle[target], 3)),
        ("SNR", np.round(fit.snr[target], 1)),
    ]
    mixture = fit.gaussian_mixture(target, consistent=True)
    if mixture is None or target not in fit.confusion:
        rejected = fit.gaussian_mixture(target)
        if rejected is not None:
            rows.append(
                (
                    "Gaussian Mixture",
                    (
                        "not used, inconsistent with the data (distance "
                        f"{rejected.distance:.3g} > {rejected.tolerance:.3g})"
                    ),
                )
            )
            rows.append(_excluded_row(fit.core_region(target)))
        rows += [
            ("Threshold", np.round(fit.threshold[target], 6)),
            ("Readout Fidelity", np.round(fit.fidelity[target], 3)),
            ("Assignment Fidelity", np.round(fit.assignment_fidelity[target], 3)),
            (
                "Effective Qubit Temperature [K]",
                temperature(fit.effective_temperature[target]),
            ),
        ]
        return [name for name, _ in rows], [value for _, value in rows]

    def compare(naive, gaussian, digits=4):
        return f"{naive:.{digits}g} / {gaussian:.{digits}g}"

    naive, gaussian = fit.confusion[target], mixture.errors()
    rows += [
        (
            "Threshold (naive / Gaussian)",
            compare(fit.threshold[target], mixture.threshold, 6),
        ),
        ("P(1|0) (naive / Gaussian)", compare(naive[0], gaussian[0])),
        ("P(0|1) (naive / Gaussian)", compare(naive[1], gaussian[1])),
        (
            "Readout Fidelity (naive / Gaussian)",
            compare(1 - sum(naive), 1 - sum(gaussian)),
        ),
        (
            "Assignment Fidelity (naive / Gaussian)",
            compare(1 - sum(naive) / 2, 1 - sum(gaussian) / 2),
        ),
        _excluded_row(fit.core_region(target)),
        ("Excited Population, Ground Preparation", f"{mixture.spurious[0]:.4g}"),
        ("Ground Population, Excited Preparation", f"{mixture.spurious[1]:.4g}"),
        (
            "Effective Qubit Temperature [K] (naive / Gaussian)",
            " / ".join(
                temperature(value)
                for value in (
                    fit.effective_temperature[target],
                    fit.mixture_effective_temperature[target],
                )
            ),
        ),
    ]
    return [name for name, _ in rows], [value for _, value in rows]


def _excluded_row(region: CoreRegion | None) -> tuple[str, str]:
    """Report the shots excluded from the Gaussian mixture fit."""
    name = "Excluded Shots (ground / excited preparation)"
    if region is None:
        return name, "-"
    counts = " / ".join(str(outside) for outside in region.outside)
    if region.excluded:
        return name, counts
    return name, f"none ({counts} outside, above the {MAX_EXCLUDED:.0%} limit)"


def _update(
    results: SingleShotClassificationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    # prefer the quantities of the Gaussian mixture fit, excluding the spurious
    # populations, when available
    mixture = results.gaussian_mixture(target, consistent=True)
    if mixture is not None:
        threshold = mixture.threshold
        fidelity = 1 - sum(mixture.errors())
        temperature = results.mixture_effective_temperature[target][0]
    else:
        threshold = results.threshold[target]
        fidelity = results.fidelity[target]
        temperature = results.effective_temperature[target][0]
    update.iq_angle(results.rotation_angle[target], platform, target)
    update.threshold(threshold, platform, target)
    update.mean_gnd_states(results.mean_gnd_states[target], platform, target)
    update.mean_exc_states(results.mean_exc_states[target], platform, target)
    update.readout_fidelity(fidelity, platform, target)
    platform.calibration.single_qubits[
        target
    ].readout.effective_temperature = temperature


single_shot_classification = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit classification routine object."""
