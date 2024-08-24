from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import (
    effective_qubit_temperature,
    format_error_single_cell,
    round_report,
    table_dict,
    table_html,
)


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
    Lambda_M: dict[QubitId, float]
    "Mapping between a given initial state to an outcome after the measurement"
    Lambda_M2: dict[QubitId, float]
    "Mapping between the outcome after the measurement and it still being that outcame after another measurement"


ReadoutCharacterizationType = np.dtype(
    [
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for ReadoutCharacterization."""


@dataclass
class ReadoutCharacterizationData(Data):
    """ReadoutCharacterization acquisition outputs."""

    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""

    delay: float = 0
    """Delay between readouts [ns]."""
    data: dict[tuple, npt.NDArray[ReadoutCharacterizationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""
    samples: dict[tuple, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: ReadoutCharacterizationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ReadoutCharacterizationData:
    """Data acquisition for resonator spectroscopy."""

    data = ReadoutCharacterizationData(
        qubit_frequencies={
            qubit: platform.qubits[qubit].drive_frequency for qubit in targets
        },
        delay=float(params.delay),
    )

    # FIXME: ADD 1st measurament and post_selection for accurate state preparation ?

    for state in [0, 1]:
        # Define the pulse sequences
        if state == 1:
            RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in targets:
            start = 0
            if state == 1:
                RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                sequence.add(RX_pulses[qubit])
                start = RX_pulses[qubit].finish
            ro_pulses[qubit] = []
            for _ in range(2):
                ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
                start += ro_pulse.duration + int(
                    params.delay
                )  # device required conversion
                sequence.add(ro_pulse)
                ro_pulses[qubit].append(ro_pulse)

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            ),
        )
        results_samples = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
            ),
        )

        # Save the data
        for qubit in targets:
            for i, ro_pulse in enumerate(ro_pulses[qubit]):
                result = results[ro_pulse.serial]
                data.register_qubit(
                    ReadoutCharacterizationType,
                    (qubit, state, i),
                    dict(i=result.voltage_i, q=result.voltage_q),
                )
                result_samples = results_samples[ro_pulse.serial]
                data.samples[qubit, state, i] = result_samples.samples.tolist()

    return data


def _fit(data: ReadoutCharacterizationData) -> ReadoutCharacterizationResults:
    """Post-processing function for ReadoutCharacterization."""

    qubits = data.qubits
    assignment_fidelity = {}
    fidelity = {}
    effective_temperature = {}
    qnd = {}
    Lambda_M = {}
    Lambda_M2 = {}
    for qubit in qubits:
        # 1st measurement (m=1)
        m1_state_1 = data.samples[qubit, 1, 0]
        nshots = len(m1_state_1)
        # state 1
        state1_count_1_m1 = np.count_nonzero(m1_state_1)
        state0_count_1_m1 = nshots - state1_count_1_m1

        m1_state_0 = data.samples[qubit, 0, 0]
        # state 0
        state1_count_0_m1 = np.count_nonzero(m1_state_0)
        state0_count_0_m1 = nshots - state1_count_0_m1

        # 2nd measurement (m=2)
        m2_state_1 = data.samples[qubit, 1, 1]
        # state 1
        state1_count_1_m2 = np.count_nonzero(m2_state_1)
        state0_count_1_m2 = nshots - state1_count_1_m2

        m2_state_0 = data.samples[qubit, 0, 1]
        # state 0
        state1_count_0_m2 = np.count_nonzero(m2_state_0)
        state0_count_0_m2 = nshots - state1_count_0_m2

        # Repeat Lambda and fidelity for each measurement ?
        Lambda_M[qubit] = [
            [state0_count_0_m1 / nshots, state0_count_1_m1 / nshots],
            [state1_count_0_m1 / nshots, state1_count_1_m1 / nshots],
        ]

        # Repeat Lambda and fidelity for each measurement ?
        Lambda_M2[qubit] = [
            [state0_count_0_m2 / nshots, state0_count_1_m2 / nshots],
            [state1_count_0_m2 / nshots, state1_count_1_m2 / nshots],
        ]

        assignment_fidelity[qubit] = (
            1 - (state1_count_0_m1 / nshots + state0_count_1_m1 / nshots) / 2
        )

        fidelity[qubit] = 2 * assignment_fidelity[qubit] - 1

        # QND FIXME: Careful revision
        P_0o_m0_1i = state0_count_1_m1 * state0_count_0_m2 / nshots**2
        P_0o_m1_1i = state1_count_1_m1 * state0_count_1_m2 / nshots**2
        P_0o_1i = P_0o_m0_1i + P_0o_m1_1i

        P_1o_m0_0i = state0_count_0_m1 * state1_count_0_m2 / nshots**2
        P_1o_m1_0i = state1_count_0_m1 * state1_count_1_m2 / nshots**2
        P_1o_0i = P_1o_m0_0i + P_1o_m1_0i

        qnd[qubit] = 1 - (P_0o_1i + P_1o_0i) / 2
        effective_temperature[qubit] = effective_qubit_temperature(
            prob_1=state0_count_1_m1 / nshots,
            prob_0=state0_count_0_m1 / nshots,
            qubit_frequency=data.qubit_frequencies[qubit],
            nshots=nshots,
        )

    return ReadoutCharacterizationResults(
        fidelity, assignment_fidelity, qnd, effective_temperature, Lambda_M, Lambda_M2
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
                    x=shots.i,
                    y=shots.q,
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
                z=fit.Lambda_M[target],
                x=["0", "1"],
                y=["0", "1"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.Lambda_M2[target],
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
    results: ReadoutCharacterizationResults, platform: Platform, target: QubitId
):
    update.readout_fidelity(results.fidelity[target], platform, target)
    update.assignment_fidelity(results.assignment_fidelity[target], platform, target)


readout_characterization = Routine(_acquisition, _fit, _plot, _update)
"""ReadoutCharacterization Routine object."""
