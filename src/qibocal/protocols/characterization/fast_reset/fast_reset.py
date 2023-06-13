from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

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

    optimal_integration_weights: Dict[Union[str, int], float]
    """
    Optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.
    """


class FastResetData(DataUnits):
    """FastReset acquisition outputs."""

    def __init__(self):
        super().__init__(
            "data",
            options=["probability", "qubit", "state", "iteration", "fast_reset"],
        )


def _acquisition(
    params: FastResetParameters, platform: Platform, qubits: Qubits
) -> FastResetData:
    """Data acquisition for resonator spectroscopy."""

    data = FastResetData()

    RX_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        sequence.add(RX_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # execute the pulse sequence
    results = platform.execute_pulse_sequence(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            fast_reset=True,
        ),
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = {
            "sample": results[ro_pulse.serial].samples,
            "qubit": [ro_pulse.qubit] * params.nshots,
            "iteration": np.arange(params.nshots),
            "state": [1] * params.nshots,
            "fast_reset": ["True"] * params.nshots,
        }
        data.add_data_from_dict(r)

    results = platform.execute_pulse_sequence(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            fast_reset=False,
        ),
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = {
            "sample": results[ro_pulse.serial].samples,
            "qubit": [ro_pulse.qubit] * params.nshots,
            "iteration": np.arange(params.nshots),
            "state": [1] * params.nshots,
            "fast_reset": ["False"] * params.nshots,
        }
        data.add_data_from_dict(r)

    # # Is it useful to inspect state 0 ?
    # ro_pulses = {}
    # sequence = PulseSequence()
    # for qubit in qubits:
    #     ro_pulses[qubit] = platform.create_qubit_readout_pulse(
    #         qubit, start=0
    #     )
    #     sequence.add(ro_pulses[qubit])

    # # execute the pulse sequence
    # results = platform.execute_pulse_sequence(
    #     sequence,
    #     ExecutionParameters(
    #         nshots=params.nshots,
    #         relaxation_time=params.relaxation_time,
    #         fast_reset = True,
    #     ),
    # )

    # # retrieve and store the results for every qubit
    # for ro_pulse in ro_pulses.values():
    #     r = {
    #         "sample":results[ro_pulse.serial].samples,
    #         "qubit": [ro_pulse.qubit] * params.nshots,
    #         "iteration": np.arange(params.nshots),
    #         "state": [0] * params.nshots,
    #         "fast_reset": ["True"] * params.nshots,
    #     }
    #     data.add_data_from_dict(r)

    # results = platform.execute_pulse_sequence(
    #     sequence,
    #     ExecutionParameters(
    #         nshots=params.nshots,
    #         relaxation_time=params.relaxation_time,
    #         fast_reset = False,
    #     ),
    # )

    # # retrieve and store the results for every qubit
    # for ro_pulse in ro_pulses.values():
    #     r = {
    #         "sample":results[ro_pulse.serial].samples,
    #         "qubit": [ro_pulse.qubit] * params.nshots,
    #         "iteration": np.arange(params.nshots),
    #         "state": [0] * params.nshots,
    #         "fast_reset": ["False"] * params.nshots,
    #     }
    #     data.add_data_from_dict(r)

    return data


def _fit(data: FastResetData) -> FastResetResults:
    """Post-processing function for FastReset."""

    # Getting some kind of fidelity

    return FastResetResults({})


def _plot(data: FastResetData, fit: FastResetResults, qubit):
    """Plotting function for FastReset."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???

    figures = []
    fig = go.Figure()

    fitting_report = ""

    # qubit_data = data.df[data.df["qubit"] == qubit]

    # state0_data = qubit_data[data.df["state"] == 0].drop(
    #     columns=["MSR", "phase", "qubit"]
    # )
    # state1_data = qubit_data[data.df["state"] == 1].drop(
    #     columns=["MSR", "phase", "qubit"]
    # )

    iterations = data.df["iteration"]
    truncate_index = data.df.fast_reset[data.df.fast_reset == "False"].index.tolist()

    fr_df = data.df.truncate(after=truncate_index[0] - 1)
    fr_states = fr_df["sample"]

    nfr_df = data.df.truncate(before=truncate_index[0])
    nfr_states = nfr_df["sample"]

    state0_count = fr_states.value_counts()[0]
    state1_count = fr_states.value_counts()[1]
    error = (state1_count - state0_count) / len(fr_states)

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=fr_states,
            mode="markers",
            name=f"q{qubit}/r_fast_reset",
            showlegend=True,
            legendgroup=f"q{qubit}/r_fast_reset",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=nfr_states,
            mode="markers",
            name=f"q{qubit}/r_no_fast_reset",
            showlegend=True,
            legendgroup=f"q{qubit}/r_no_fast_reset",
        ),
    )

    fitting_report += f"q{qubit}/r | state0 count: {state0_count:.0f}<br>"
    fitting_report += f"q{qubit}/r | state1 count: {state1_count:.0f}<br>"
    fitting_report += f"q{qubit}/r | Error: {error:.6f}<br>"
    fitting_report += f"q{qubit}/r | Fidelity(add 0 to 1 error): {(1 - error):.6f}<br>"

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Shot",
        yaxis_title="State",
    )

    figures.append(fig)

    return figures, fitting_report


fast_reset = Routine(_acquisition, _fit, _plot)
"""FastReset Routine object."""
