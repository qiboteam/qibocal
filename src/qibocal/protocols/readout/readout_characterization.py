from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
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

    assignment_fidelity: dict[QubitId, float]
    """Assignment fidelity."""
    qnd: dict[QubitId, float]
    "QND-ness of the measurement"
    qnd_pi: dict[QubitId, float]
    "QND-ness of the measurement"
    effective_temperature: dict[QubitId, list[float]]
    """Effective qubit temperature."""

    @property
    def readout_fidelity(self):
        return {qubit: 2 * fid - 1 for qubit, fid in self.assignment_fidelity.items()}


@dataclass
class ReadoutCharacterizationData(Data):
    """ReadoutCharacterization acquisition outputs."""

    nshots: int
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    delay: float = 0
    """Delay between readouts [ns]."""

    angle: dict[QubitId, float] = field(default_factory=dict)
    threshold: dict[QubitId, float] = field(default_factory=dict)
    data: dict[tuple, np.ndarray] = field(default_factory=dict)


def readout_sequence(platform, delay, qubit, state):
    natives = platform.natives.single_qubit[qubit]

    sequence = PulseSequence()
    if state == 1:
        sequence = natives.RX()

    for _ in range(2):
        sequence |= natives.MZ()
        sequence.append((platform.qubits[qubit].acquisition, Delay(duration=delay)))

    sequence |= natives.RX()
    sequence |= natives.MZ()
    return sequence


def _acquisition(
    params: ReadoutCharacterizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutCharacterizationData:
    """Data acquisition for resonator spectroscopy."""

    data = ReadoutCharacterizationData(
        qubit_frequencies={
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
        nshots=params.nshots,
    )

    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += readout_sequence(
                platform=platform,
                delay=params.delay,
                qubit=qubit,
                state=state,
            )

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
    qnd = {}
    qnd_pi = {}
    effective_temperature = {}

    for qubit in qubits:
        shots = {0: [], 1: []}
        for state in range(2):
            for m in range(3):
                shots[state].append(
                    classify(
                        data.data[qubit, state, m],
                        data.angle[qubit],
                        data.threshold[qubit],
                    )
                )

        assignment_fidelity[qubit] = compute_assignment_fidelity(
            shots[1][0], shots[0][0]
        )
        qnd[qubit], qnd_pi[qubit] = compute_qnd(shots[0], shots[1])
        effective_temperature[qubit] = effective_qubit_temperature(
            predictions=shots[0][0],
            qubit_frequency=data.qubit_frequencies[qubit],
            nshots=data.nshots,
        )

    return ReadoutCharacterizationResults(
        assignment_fidelity=assignment_fidelity,
        qnd=qnd,
        qnd_pi=qnd_pi,
        effective_temperature=effective_temperature,
    )


def _plot(
    data: ReadoutCharacterizationData,
    fit: ReadoutCharacterizationResults,
    target: QubitId,
):
    """Plotting function for ReadoutCharacterization."""

    figures = []
    fitting_report = ""
    fig = go.Figure()
    for state in range(2):
        for measure in range(3):
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
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Delay between readouts [ns]",
                    "Assignment Fidelity",
                    "QND",
                    "QND-PI",
                    "Effective Qubit Temperature [K]",
                ],
                [
                    np.round(data.delay),
                    np.round(fit.assignment_fidelity[target], 6),
                    np.round(fit.qnd[target], 6),
                    np.round(fit.qnd_pi[target], 6),
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
    update.readout_fidelity(results.readout_fidelity[target], platform, target)
    platform.calibration.single_qubits[
        target
    ].readout.effective_temperature = results.effective_temperature[target][0]


readout_characterization = Routine(_acquisition, _fit, _plot, _update)
"""ReadoutCharacterization Routine object."""
