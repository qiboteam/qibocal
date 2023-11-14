from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class ReadoutCharacterizationParameters(Parameters):
    """ReadoutCharacterization runcard inputs."""


@dataclass
class ReadoutCharacterizationResults(Results):
    """ReadoutCharacterization outputs."""

    fidelity: dict[QubitId, float]
    "Fidelity of the measurement"
    assignment_fidelity: dict[QubitId, float]
    """Assignment fidelity."""
    qnd: dict[QubitId, float]
    "QND-ness of the measurement"
    Lambda_M: dict[QubitId, float]
    "Mapping between a given initial state to an outcome adter the measurement"


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

    data: dict[tuple, npt.NDArray[ReadoutCharacterizationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""
    samples: dict[tuple, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: ReadoutCharacterizationParameters, platform: Platform, qubits: Qubits
) -> ReadoutCharacterizationData:
    """Data acquisition for resonator spectroscopy."""

    data = ReadoutCharacterizationData()

    # FIXME: ADD 1st measurament and post_selection for accurate state preparation ?

    for state in [0, 1]:
        # Define the pulse sequences
        if state == 1:
            RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            start = 0
            if state == 1:
                RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                sequence.add(RX_pulses[qubit])
                start = RX_pulses[qubit].finish
            ro_pulses[qubit] = []
            for _ in range(2):
                ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
                start += ro_pulse.duration
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

        # Save the data
        for qubit in qubits:
            for i, ro_pulse in enumerate(ro_pulses[qubit]):
                result = results[ro_pulse.serial]
                data.register_qubit(
                    ReadoutCharacterizationType,
                    (qubit, state, i),
                    dict(i=result.voltage_i, q=result.voltage_q),
                )
    for qubit, state, measure in data.data.keys():
        model = QubitFit()
        model.iq_angle = qubits[qubit].iq_angle
        model.threshold = qubits[qubit].threshold
        qubit_data = data.data[qubit, state, measure]
        predictions = model.predict(np.stack([qubit_data.i, qubit_data.q], axis=-1))
        data.samples[qubit, state, measure] = predictions.tolist()
    return data


def _fit(data: ReadoutCharacterizationData) -> ReadoutCharacterizationResults:
    """Post-processing function for ReadoutCharacterization."""

    qubits = data.qubits
    assignment_fidelity = {}
    fidelity = {}
    qnd = {}
    Lambda_M = {}
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

    return ReadoutCharacterizationResults(fidelity, assignment_fidelity, qnd, Lambda_M)


def _plot(
    data: ReadoutCharacterizationData, fit: ReadoutCharacterizationResults, qubit
):
    """Plotting function for ReadoutCharacterization."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???

    figures = []
    fitting_report = ""
    fig = go.Figure()
    for state in range(2):
        for measure in range(2):
            shots = data.data[qubit, state, measure]

            fig.add_trace(
                go.Scatter(
                    x=shots.i,
                    y=shots.q,
                    name=f"state {state} measure {measure}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3),
                )
            )
    figures.append(fig)
    if fit is not None:
        fig2 = go.Figure()

        fig2.add_trace(
            go.Heatmap(
                z=fit.Lambda_M[qubit],
            )
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                ["Assignment Fidelity", "Fidelity", "QND"],
                [
                    np.round(fit.assignment_fidelity[qubit], 6),
                    np.round(fit.fidelity[qubit], 6),
                    np.round(fit.qnd[qubit], 6),
                ],
            )
        )
        figures.append(fig2)

    return figures, fitting_report


def _update(
    results: ReadoutCharacterizationResults, platform: Platform, qubit: QubitId
):
    update.readout_fidelity(results.fidelity[qubit], platform, qubit)
    update.assignment_fidelity(results.assignment_fidelity[qubit], platform, qubit)


readout_characterization = Routine(_acquisition, _fit, _plot, _update)
"""ReadoutCharacterization Routine object."""
