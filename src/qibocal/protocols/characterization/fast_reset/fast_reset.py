from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

# TODO: IBM Fast Reset until saturation loop
# https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/reset/backend_reset


@dataclass
class FastResetParameters(Parameters):
    """FastReset runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


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

    data: dict[tuple[QubitId, int, bool], npt.NDArray[FastResetType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, probability, state, fast_reset):
        """Store output for single qubit."""
        ar = np.empty(probability.shape, dtype=FastResetType)
        ar["probability"] = probability
        self.data[qubit, state, fast_reset] = np.rec.array(ar)


def _acquisition(
    params: FastResetParameters, platform: Platform, qubits: Qubits
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
            for qubit in qubits:
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
                    qubit,
                    probability=result.samples,
                    state=state,
                    fast_reset=fast_reset,
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
        error_fr1 = 1 - (nshots - state0_count_1fr) / nshots

        state1_count_1nfr = np.count_nonzero(nfr_states)
        state0_count_1nfr = nshots - state1_count_1nfr
        error_nfr1 = 1 - (nshots - state0_count_1nfr) / nshots

        # state 0
        fr_states = data[qubit, 0, True].probability
        nfr_states = data[qubit, 0, False].probability

        state1_count_0fr = np.count_nonzero(fr_states)
        state0_count_0fr = nshots - state1_count_0fr
        error_fr0 = (nshots - state0_count_0fr) / nshots

        state1_count_0nfr = np.count_nonzero(nfr_states)
        state0_count_0nfr = nshots - state1_count_0nfr
        error_nfr0 = (nshots - state0_count_0nfr) / nshots

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


def _plot(data: FastResetData, fit: FastResetResults, qubit):
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
            "Relaxation Time",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            z=fit.Lambda_M_fr[qubit],
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=f"{qubit}: Fast Reset",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="State", row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1])
    fig.update_xaxes(tickvals=[0, 1])

    fig.add_trace(
        go.Heatmap(
            z=fit.Lambda_M_nfr[qubit],
            coloraxis="coloraxis",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(coloraxis={"colorscale": "viridis"})

    fig.update_xaxes(
        title_text="State prepared",
        row=1,
        col=2,
    )

    fitting_report += (
        f"{qubit} | Fidelity [Fast Reset]: {fit.fidelity_fr[qubit]:.6f}<br>"
    )
    fitting_report += (
        f"{qubit}| Fidelity [Relaxation Time]: {fit.fidelity_nfr[qubit]:.6f}<br>"
    )

    # last part
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="State prepared",
        yaxis_title="State read",
    )

    fig.update_xaxes(tickvals=[0, 1])
    fig.update_yaxes(tickvals=[0, 1])

    figures.append(fig)

    return figures, fitting_report


fast_reset = Routine(_acquisition, _fit, _plot)
"""FastReset Routine object."""
