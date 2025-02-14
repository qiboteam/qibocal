from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import (
    evaluate_grid,
    format_error_single_cell,
    plot_results,
    round_report,
    table_dict,
    table_html,
)

ROC_LENGHT = 800
ROC_WIDTH = 800
DEFAULT_CLASSIFIER = "qubit_fit"


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64), ("state", int)])
"""Custom dtype for rabi amplitude."""


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""

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


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: Platform,
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

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequences, all_ro_pulses = [], []
    for state in [0, 1]:
        sequence = PulseSequence()
        RX_pulses = {}
        ro_pulses = {}
        for qubit in targets:
            RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                qubit, start=RX_pulses[qubit].finish
            )
            if state == 1:
                sequence.add(RX_pulses[qubit])
            sequence.add(ro_pulses[qubit])

        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)
    else:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    data = SingleShotClassificationData(
        nshots=params.nshots,
        qubit_frequencies={
            qubit: platform.qubits[qubit].drive_frequency for qubit in targets
        },
    )
    for ig, (state, ro_pulses) in enumerate(zip([0, 1], all_ro_pulses)):
        for qubit in targets:
            serial = ro_pulses[qubit].serial
            if params.unrolling:
                result = results[serial][ig]
            else:
                result = results[ig][serial]
            data.register_qubit(
                ClassificationType,
                (qubit),
                dict(
                    i=result.voltage_i,
                    q=result.voltage_q,
                    state=[state] * params.nshots,
                ),
            )

    return data


def train_classifier(data, qubit):
    qubit_data = data.data[qubit]
    i_values = qubit_data["i"]
    q_values = qubit_data["q"]
    states = qubit_data["state"]
    model = QubitFit()
    model.fit(i_values, q_values, states)
    return model


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits

    threshold = {}
    rotation_angle = {}
    mean_gnd_states = {}
    mean_exc_states = {}
    fidelity = {}
    assignment_fidelity = {}
    grid_preds_dict = {}
    effective_temperature = {}
    for qubit in qubits:
        qubit_data = data.data[qubit]
        i_values = qubit_data["i"]
        q_values = qubit_data["q"]
        iq_values = np.stack((i_values, q_values), axis=-1)
        states = qubit_data["state"]
        model = train_classifier(data, qubit)
        grid = evaluate_grid(qubit_data)
        grid_preds = model.predict(grid)
        threshold[qubit] = model.threshold
        rotation_angle[qubit] = model.angle
        mean_gnd_states[qubit] = model.iq_mean0.tolist()
        mean_exc_states[qubit] = model.iq_mean1.tolist()
        fidelity[qubit] = model.fidelity
        assignment_fidelity[qubit] = model.assignment_fidelity
        iq_state0 = iq_values[states == 0]
        predictions_state0 = model.predict(iq_state0.tolist())
        effective_temperature[qubit] = model.effective_temperature(
            predictions_state0, data.qubit_frequencies[qubit]
        )
        grid_preds_dict[qubit] = grid_preds.tolist()
    return SingleShotClassificationResults(
        threshold=threshold,
        rotation_angle=rotation_angle,
        mean_gnd_states=mean_gnd_states,
        mean_exc_states=mean_exc_states,
        fidelity=fidelity,
        assignment_fidelity=assignment_fidelity,
        effective_temperature=effective_temperature,
        grid_preds=grid_preds_dict,
    )


def _plot(
    data: SingleShotClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    fitting_report = ""
    figures = plot_results(data, target, 2, fit)
    if fit is not None:
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
                    "Effective Qubit Temperature [K]",
                ],
                [
                    np.round(fit.mean_gnd_states[target], 3),
                    np.round(fit.mean_exc_states[target], 3),
                    np.round(fit.rotation_angle[target], 3),
                    np.round(fit.threshold[target], 6),
                    np.round(fit.fidelity[target], 3),
                    np.round(fit.assignment_fidelity[target], 3),
                    format_error_single_cell(
                        round_report([fit.effective_temperature[target]])
                    ),
                ],
            )
        )

    return figures, fitting_report


def _update(
    results: SingleShotClassificationResults, platform: Platform, target: QubitId
):
    update.iq_angle(results.rotation_angle[target], platform, target)
    update.threshold(results.threshold[target], platform, target)
    update.mean_gnd_states(results.mean_gnd_states[target], platform, target)
    update.mean_exc_states(results.mean_exc_states[target], platform, target)
    update.readout_fidelity(results.fidelity[target], platform, target)
    update.assignment_fidelity(results.assignment_fidelity[target], platform, target)


single_shot_classification = Routine(_acquisition, _fit, _plot, _update)
"""Qubit classification routine object."""
