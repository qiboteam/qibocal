from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, PulseSequence

from qibocal import update
from qibocal.auto.operation import (
    DATAFILE,
    RESULTSFILE,
    Data,
    Parameters,
    Protocol,
    QubitId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    effective_qubit_temperature,
    format_error_single_cell,
    linear_classifier,
    project,
    round_report,
    table_dict,
    table_html,
)

from .plot import plot_results

__all__ = [
    "ClassificationType",
    "SingleShotClassificationData",
    "SingleShotClassificationParameters",
    "single_shot_classification",
]

LEGACY_FIELDS = {
    "classifiers_list",
    "savedir",
    "names",
    "y_preds",
    "grid_preds",
    "models",
    "benchmark_table",
    "classifiers_hpars",
    "x_tests",
    "y_tests",
}
"""Fields of the removed classifiers, still found in runcards and saved reports."""


def _known(cls: type, values: dict) -> dict:
    """Drop the fields of the removed classifiers, which `cls` does not define."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in names or k not in LEGACY_FIELDS}


def evaluate_snr(zeros: npt.NDArray, ones: npt.NDArray) -> float:
    r"""Compute the single-shot SNR of the two clouds.

    The shots are projected on the line joining the two means, and the SNR is the
    separation of the means over the single-shot noise on their difference,
    :math:`|\mu_1 - \mu_0| / \sqrt{\sigma_0^2 + \sigma_1^2}`.
    """
    line = np.mean(ones, axis=0) - np.mean(zeros, axis=0)
    projection_zeros, projection_ones = np.dot(zeros, line), np.dot(ones, line)
    mu0, std0 = np.mean(projection_zeros), np.std(projection_zeros)
    mu1, std1 = np.mean(projection_ones), np.std(projection_ones)
    return np.abs(mu1 - mu0) / np.sqrt(std0**2 + std1**2)


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    @classmethod
    def load(cls, input_parameters):
        """Load parameters, ignoring the ones of the removed classifiers."""
        legacy = LEGACY_FIELDS & set(input_parameters)
        if legacy:
            log.warning(
                f"Ignoring {sorted(legacy)}: the classifiers have been removed, and "
                "the states are classified by a linear threshold."
            )
        return super().load(_known(cls, input_parameters))


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64), ("state", int)])
"""Custom dtype for classification."""


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @classmethod
    def load(cls, path: Path, filename: str = DATAFILE):
        """Load data, also saved with the removed classifiers' fields."""
        params = cls.load_params(path, filename)
        data = cls.load_data(path, filename)
        if params is None and data is None:
            return None
        params = _known(cls, params or {})
        return cls(**params) if data is None else cls(data=data, **params)

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

    threshold: dict[QubitId, float] = field(default_factory=dict)
    """Threshold for classification, along the rotated I axis."""
    rotation_angle: dict[QubitId, float] = field(default_factory=dict)
    """Angle rotating the axis connecting the centroids to the I axis."""
    mean_gnd_states: dict[QubitId, list[float]] = field(default_factory=dict)
    """Centroid of the ground state blob."""
    mean_exc_states: dict[QubitId, list[float]] = field(default_factory=dict)
    """Centroid of the excited state blob."""
    fidelity: dict[QubitId, float] = field(default_factory=dict)
    """Readout fidelity."""
    assignment_fidelity: dict[QubitId, float] = field(default_factory=dict)
    """Assignment fidelity."""
    effective_temperature: dict[QubitId, float] = field(default_factory=dict)
    """Qubit effective temperature from Boltzmann distribution."""
    snr: dict[QubitId, float] = field(default_factory=dict)
    """SNR for two clouds"""

    @classmethod
    def load(cls, path: Path, filename: str = RESULTSFILE):
        """Load results, also saved with the removed classifiers' fields."""
        params = cls.load_params(path, filename)
        return None if params is None else cls(**_known(cls, params))


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SingleShotClassificationData:
    """Measure the qubits prepared in the ground and excited states.

    Example:
        .. code-block:: yaml

            - id: single_shot_classification_1
              operation: single_shot_classification
              parameters:
                nshots: 5000
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
    results = SingleShotClassificationResults()
    for qubit in data.qubits:
        ground, excited = data.state_zero(qubit), data.state_one(qubit)
        angle, threshold = linear_classifier(ground, excited)
        ground_x, excited_x = project(ground, angle), project(excited, angle)
        # probabilities of measuring a state, given the prepared one
        p10 = float(np.mean(ground_x > threshold))
        p01 = float(np.mean(excited_x <= threshold))
        results.rotation_angle[qubit] = angle
        results.threshold[qubit] = threshold
        results.mean_gnd_states[qubit] = np.mean(ground, axis=0).tolist()
        results.mean_exc_states[qubit] = np.mean(excited, axis=0).tolist()
        results.fidelity[qubit] = 1 - p10 - p01
        results.assignment_fidelity[qubit] = 1 - (p10 + p01) / 2
        results.snr[qubit] = evaluate_snr(ground, excited)
        results.effective_temperature[qubit] = effective_qubit_temperature(
            prob_0=1 - p10,
            prob_1=p10,
            qubit_frequency=data.qubit_frequencies[qubit],
            nshots=len(ground),
        )
    return results


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    figures = plot_results(data, target, 2, fit)
    if fit is None or target not in fit:
        return figures, ""
    fitting_report = table_html(
        table_dict(
            target,
            [
                "Average State 0",
                "Average State 1",
                "Rotational Angle",
                "Threshold",
                "Readout Fidelity",
                "Assignment Fidelity",
                "SNR",
                "Effective Qubit Temperature [K]",
            ],
            [
                np.round(fit.mean_gnd_states[target], 3),
                np.round(fit.mean_exc_states[target], 3),
                np.round(fit.rotation_angle[target], 3),
                np.round(fit.threshold[target], 6),
                np.round(fit.fidelity[target], 3),
                np.round(fit.assignment_fidelity[target], 3),
                np.round(fit.snr[target], 2),
                format_error_single_cell(
                    round_report([fit.effective_temperature[target]])
                ),
            ],
        )
    )
    return figures, fitting_report


def _update(
    results: SingleShotClassificationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.iq_angle(results.rotation_angle[target], platform, target)
    update.threshold(results.threshold[target], platform, target)
    update.mean_gnd_states(results.mean_gnd_states[target], platform, target)
    update.mean_exc_states(results.mean_exc_states[target], platform, target)
    update.readout_fidelity(results.fidelity[target], platform, target)
    platform.calibration.single_qubits[
        target
    ].readout.effective_temperature = results.effective_temperature[target][0]


single_shot_classification = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit classification routine object."""
