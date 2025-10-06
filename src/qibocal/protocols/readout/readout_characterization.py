from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, Delay, PulseSequence, Readout

from ... import update
from ...auto.operation import Data, Parameters, QubitId, Results, Routine
from ...calibration import CalibrationPlatform
from ..utils import (
    classify,
    compute_assignment_fidelity,
    compute_qnd,
    effective_qubit_temperature,
    format_error_single_cell,
    round_report,
    table_dict,
    table_html,
)

__all__ = ["readout_characterization"]


@dataclass
class ReadoutCharacterizationParameters(Parameters):
    """ReadoutCharacterization runcard inputs."""

    delay: float = 0
    """Delay between readouts, could account for resonator deplation or not [ns]."""


@dataclass
class ReadoutCharacterizationResults(Results):
    """ReadoutCharacterization outputs."""

    fidelity: dict[QubitId, float]
    "Fidelity of the measurement"
    assignment_fidelity: dict[QubitId, float]
    """Assignment fidelity."""
    qnd: dict[QubitId, float]
    "QND-ness of the measurement"
    effective_temperature: dict[QubitId, list[float]]
    """Effective qubit temperature."""
    lambda_m: dict[QubitId, float]
    "Mapping between a given initial state to an outcome after the measurement"
    lambda_m2: dict[QubitId, float]
    "Mapping between the outcome after the measurement and it still being that outcame after another measurement"


@dataclass
class ReadoutCharacterizationData(Data):
    """ReadoutCharacterization acquisition outputs."""

    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    delay: float = 0
    """Delay between readouts [ns]."""

    angle: dict[QubitId, float] = field(default_factory=dict)
    threshold: dict[QubitId, float] = field(default_factory=dict)
    data: dict[tuple, np.ndarray] = field(default_factory=dict)


def _acquisition(
    params: ReadoutCharacterizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutCharacterizationData:
    """Data acquisition for resonator spectroscopy."""

    data = ReadoutCharacterizationData(
        qubit_frequencies={
            # TODO: should this be the drive frequency instead?
            qubit: float(platform.calibration.single_qubits[qubit].qubit.frequency_01)
            for qubit in targets
        },
        angle={
            qubit: platform.config(platform.qubits[qubit].acquisition).iq_angle
            for qubit in targets
        },
        threshold={
            qubit: platform.config(platform.qubits[qubit].acquisition).threshold
            for qubit in targets
        },
        delay=float(params.delay),
    )

    # FIXME: ADD 1st measurament and post_selection for accurate state preparation ?

    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            ro_channel = natives.MZ()[0][0]
            if state == 1:
                sequence += natives.RX()
            sequence.append((ro_channel, Delay(duration=natives.RX()[0][1].duration)))
            sequence += natives.MZ()
            sequence.append((ro_channel, Delay(duration=params.delay)))
            sequence += natives.MZ()

        # execute the pulse sequence
        results = platform.execute(
            [sequence],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )

        # Save the data
        for qubit in targets:
            readouts = [
                pulse
                for pulse in sequence.channel(platform.qubits[qubit].acquisition)
                if isinstance(pulse, Readout)
            ]
            for j, ro_pulse in enumerate(readouts):
                data.data[qubit, state, j] = results[ro_pulse.id]
    return data


def _fit(data: ReadoutCharacterizationData) -> ReadoutCharacterizationResults:
    """Post-processing function for ReadoutCharacterization."""

    qubits = data.qubits
    assignment_fidelity = {}
    fidelity = {}
    effective_temperature = {}
    qnd = {}
    lambda_m, lambda_m2 = {}, {}
    for qubit in qubits:
        m1_state_1 = classify(
            data.data[qubit, 1, 0], data.angle[qubit], data.threshold[qubit]
        )
        m1_state_0 = classify(
            data.data[qubit, 0, 0], data.angle[qubit], data.threshold[qubit]
        )
        m2_state_1 = classify(
            data.data[qubit, 1, 1], data.angle[qubit], data.threshold[qubit]
        )
        m2_state_0 = classify(
            data.data[qubit, 0, 1], data.angle[qubit], data.threshold[qubit]
        )

        assignment_fidelity[qubit] = compute_assignment_fidelity(m1_state_1, m1_state_0)
        qnd[qubit], lambda_m[qubit], lambda_m2[qubit] = compute_qnd(
            m1_state_1, m1_state_0, m2_state_1, m2_state_0
        )

        fidelity[qubit] = 2 * assignment_fidelity[qubit] - 1

        prob_1 = np.mean(m1_state_0)
        effective_temperature[qubit] = effective_qubit_temperature(
            prob_1=prob_1,
            prob_0=1 - prob_1,
            qubit_frequency=data.qubit_frequencies[qubit],
            nshots=len(m1_state_0),
        )

    return ReadoutCharacterizationResults(
        fidelity, assignment_fidelity, qnd, effective_temperature, lambda_m, lambda_m2
    )


def _plot(
    data: ReadoutCharacterizationData,
    fit: ReadoutCharacterizationResults,
    target: QubitId,
):
    """Plotting function for ReadoutCharacterization."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???

    figures = []
    fitting_report = ""
    fig = go.Figure()
    for state in range(2):
        for measure in range(2):
            shots = data.data[target, state, measure]

            fig.add_trace(
                go.Scatter(
                    x=shots[:, 0],
                    y=shots[:, 1],
                    name=f"Prepared state {state} measurement {measure}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3),
                )
            )
    fig.update_layout(
        title={
            "text": "IQ Plane",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="I",
        yaxis_title="Q",
    )

    figures.append(fig)
    if fit is not None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "1st measurement statistics",
                "2nd measurement statistics",
            ),
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.lambda_m[target],
                x=["0", "1"],
                y=["0", "1"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.lambda_m2[target],
                x=["0", "1"],
                y=["0", "1"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Measured state", row=1, col=1)
        fig.update_xaxes(title_text="Measured state", row=1, col=2)
        fig.update_yaxes(title_text="Prepared state", row=1, col=1)
        fig.update_yaxes(title_text="Prepared state", row=1, col=2)

        figures.append(fig)

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Delay between readouts [ns]",
                    "Assignment Fidelity",
                    "Fidelity",
                    "QND",
                    "Effective Qubit Temperature [K]",
                ],
                [
                    np.round(data.delay),
                    np.round(fit.assignment_fidelity[target], 6),
                    np.round(fit.fidelity[target], 6),
                    np.round(fit.qnd[target], 6),
                    format_error_single_cell(
                        round_report([fit.effective_temperature[target]])
                    ),
                ],
            )
        )

    return figures, fitting_report


def _update(
    results: ReadoutCharacterizationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_fidelity(results.fidelity[target], platform, target)
    platform.calibration.single_qubits[
        target
    ].readout.effective_temperature = results.effective_temperature[target][0]


readout_characterization = Routine(_acquisition, _fit, _plot, _update)
"""ReadoutCharacterization Routine object."""
