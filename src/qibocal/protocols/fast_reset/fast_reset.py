from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import table_dict, table_html

# TODO: IBM Fast Reset until saturation loop
# https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/reset/backend_reset


@dataclass
class FastResetParameters(Parameters):
    """FastReset runcard inputs."""


@dataclass
class FastResetResults(Results):
    """FastReset outputs."""

    fidelity_nfr: dict[QubitId, float]
    "Fidelity of the measurement with relaxation time"
    Lambda_M_nfr: dict[QubitId, float]
    "Mapping between a given initial state to an outcome adter the measurement with relaxation time"
    fidelity_fr: dict[QubitId, float]
    "Fidelity of the measurement with fast reset"
    Lambda_M_fr: dict[QubitId, float]
    "Mapping between a given initial state to an outcome adter the measurement with fast reset"


FastResetType = np.dtype(
    [
        ("probability", np.float64),
    ]
)
"""Custom dtype for FastReset."""


@dataclass
class FastResetData(Data):
    """FastReset acquisition outputs."""

    data: dict[tuple, npt.NDArray[FastResetType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: FastResetParameters, platform: Platform, targets: list[QubitId]
) -> FastResetData:
    """Data acquisition for resonator spectroscopy."""

    data = FastResetData()
    for state in [0, 1]:
        for fast_reset in [True, False]:
            # Define the pulse sequences
            if state == 1:
                RX_pulses = {}
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in targets:
                if state == 1:
                    RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                    ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                        qubit, start=RX_pulses[qubit].finish
                    )
                    sequence.add(RX_pulses[qubit])
                else:
                    ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                        qubit, start=0
                    )
                sequence.add(ro_pulses[qubit])

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    fast_reset=fast_reset,
                ),
            )

            # Save the data
            for ro_pulse in ro_pulses.values():
                result = results[ro_pulse.serial]
                qubit = ro_pulse.qubit
                data.register_qubit(
                    FastResetType,
                    (qubit, state, fast_reset),
                    dict(probability=result.samples),
                )

    return data


def _fit(data: FastResetData) -> FastResetResults:
    """Post-processing function for FastReset."""

    qubits = data.qubits
    fidelity_nfr = {}
    Lambda_M_nfr = {}
    fidelity_fr = {}
    Lambda_M_fr = {}
    for qubit in qubits:
        # state 1
        fr_states = data[qubit, 1, True].probability
        nfr_states = data[qubit, 1, False].probability

        nshots = len(fr_states)

        state1_count_1fr = np.count_nonzero(fr_states)
        state0_count_1fr = nshots - state1_count_1fr

        state1_count_1nfr = np.count_nonzero(nfr_states)
        state0_count_1nfr = nshots - state1_count_1nfr

        # state 0
        fr_states = data[qubit, 0, True].probability
        nfr_states = data[qubit, 0, False].probability

        state1_count_0fr = np.count_nonzero(fr_states)
        state0_count_0fr = nshots - state1_count_0fr

        state1_count_0nfr = np.count_nonzero(nfr_states)
        state0_count_0nfr = nshots - state1_count_0nfr

        # Repeat Lambda and fidelity for each measurement ?
        Lambda_M_nfr[qubit] = [
            [state0_count_0nfr / nshots, state0_count_1nfr / nshots],
            [state1_count_0nfr / nshots, state1_count_1nfr / nshots],
        ]

        # Repeat Lambda and fidelity for each measurement ?
        Lambda_M_fr[qubit] = [
            [state0_count_0fr / nshots, state0_count_1fr / nshots],
            [state1_count_0fr / nshots, state1_count_1fr / nshots],
        ]

        fidelity_nfr[qubit] = (
            1 - (state1_count_0nfr / nshots + state0_count_1nfr / nshots) / 2
        )

        fidelity_fr[qubit] = (
            1 - (state1_count_0fr / nshots + state0_count_1fr / nshots) / 2
        )

    return FastResetResults(fidelity_nfr, Lambda_M_nfr, fidelity_fr, Lambda_M_fr)


def _plot(data: FastResetData, fit: FastResetResults, target: QubitId):
    """Plotting function for FastReset."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Fast Reset",
            "Relaxation Time [ns]",
        ),
    )

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                z=fit.Lambda_M_fr[target],
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Fidelity [Fast Reset]", "Fidelity [Relaxation Time]"],
                [
                    np.round(fit.fidelity_fr[target], 6),
                    np.round(fit.fidelity_nfr[target], 6),
                ],
            )
        )

        fig.add_trace(
            go.Heatmap(
                z=fit.Lambda_M_nfr[target],
                coloraxis="coloraxis",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(
        title_text=f"{target}: Fast Reset",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="State", row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1])
    fig.update_xaxes(tickvals=[0, 1])

    fig.update_layout(coloraxis={"colorscale": "viridis"})

    fig.update_xaxes(
        title_text="State prepared",
        row=1,
        col=2,
    )

    # last part
    fig.update_layout(
        showlegend=False,
        xaxis_title="State prepared",
        yaxis_title="State measured",
    )

    figures.append(fig)

    return figures, fitting_report


fast_reset = Routine(_acquisition, _fit, _plot)
"""FastReset Routine object."""
