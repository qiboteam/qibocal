from dataclasses import dataclass, field
from typing import Dict, Optional


import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
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

    optimal_integration_weights: Dict[QubitId, float]
    """
    Optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.
    """


FastResetType = np.dtype(
    [("probability", np.float64), ("state", np.int64), ("iteration", np.int64), ("fast_reset", np.int64)]
)
"""Custom dtype for FastReset."""

@dataclass
class FastResetData(Data):
    """FastReset acquisition outputs."""

    data: dict[QubitId, npt.NDArray[FastResetType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, probability, state, iteration, fast_reset):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(probability) else probability.shape
        ar = np.empty(shape, dtype=FastResetType)
        ar["probability"] = probability
        ar["state"] = state
        ar["iteration"] = iteration
        ar["fast_reset"] = fast_reset
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


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
    # for ro_pulse in ro_pulses.values():
    #     r = {
    #         "sample": results[ro_pulse.serial].samples,
    #         "qubit": [ro_pulse.qubit] * params.nshots,
    #         "iteration": np.arange(params.nshots),
    #         "state": [1] * params.nshots,
    #         "fast_reset": ["True"] * params.nshots,
    #     }
    #     data.add_data_from_dict(r)

    for ro_pulse in ro_pulses.values():
        result = results[ro_pulse.serial]
        qubit = ro_pulse.qubit
        data.register_qubit(
            qubit,
            probability=result.samples,
            state= [1] * params.nshots,
            iteration= np.arange(params.nshots),
            fast_reset= [1] * params.nshots,
        )
    
    results = platform.execute_pulse_sequence(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            fast_reset=False,
        ),
    )
    
    # # retrieve and store the results for every qubit
    # for ro_pulse in ro_pulses.values():
    #     r = {
    #         "sample": results[ro_pulse.serial].samples,
    #         "qubit": [ro_pulse.qubit] * params.nshots,
    #         "iteration": np.arange(params.nshots),
    #         "state": [1] * params.nshots,
    #         "fast_reset": ["False"] * params.nshots,
    #     }
    #     data.add_data_from_dict(r)
        
    for ro_pulse in ro_pulses.values():
        result = results[ro_pulse.serial]
        qubit = ro_pulse.qubit
        data.register_qubit(
            qubit,
            probability=result.samples,
            state= [1] * params.nshots,
            iteration= np.arange(params.nshots),
            fast_reset= [0] * params.nshots,
        )

    # It's useful to inspect state 0 for things like active initialization
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
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
            fast_reset = True,
        ),
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        result = results[ro_pulse.serial]
        qubit = ro_pulse.qubit
        data.register_qubit(
            qubit,
            probability=result.samples,
            state= [0] * params.nshots,
            iteration = np.arange(params.nshots),
            fast_reset = [1] * params.nshots,
        )

    results = platform.execute_pulse_sequence(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            fast_reset = False,
        ),
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        result = results[ro_pulse.serial]
        qubit = ro_pulse.qubit
        data.register_qubit(
            qubit,
            probability=result.samples,
            state= [0] * params.nshots,
            iteration= np.arange(params.nshots),
            fast_reset= [0] * params.nshots,
        )

    return data


def _fit(data: FastResetData) -> FastResetResults:
    """Post-processing function for FastReset."""

    # Getting some kind of fidelity

    return FastResetResults({})


def _plot(data: FastResetData, fit: FastResetResults, qubit):
    """Plotting function for FastReset."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???

    figures = []
    fitting_report = ""
    fig = go.Figure()
    
    
    qubit_data = data[qubit]

    truncate_index_state = np.min(np.where(qubit_data.state == 0))
    state1 = qubit_data[:truncate_index_state]
    state0 = qubit_data[truncate_index_state:]
    
    # import pdb; pdb.set_trace()
    
    iterations = qubit_data.iteration  
    
    truncate_index = np.min(np.where(state1.fast_reset == 0))
    fr_states = state1.probability[:truncate_index]
    nfr_states = state1.probability[truncate_index:]

    unique, counts = np.unique(fr_states, return_counts=True)
    state0_count_1fr = counts[0]
    state1_count_1fr = counts[1]
    error_fr1 = (state1_count_1fr - state0_count_1fr) / len(fr_states)
    
    unique, counts = np.unique(nfr_states, return_counts=True)
    state0_count_1nfr = counts[0]
    state1_count_1nfr = counts[1]
    error_nfr1 = (state1_count_1nfr - state0_count_1nfr) / len(fr_states)
    
    truncate_index = np.min(np.where(state0.fast_reset == 0))
    fr_states = state0.probability[:truncate_index]
    nfr_states = state0.probability[truncate_index:]

    unique, counts = np.unique(fr_states, return_counts=True)
    state0_count_0fr = counts[0]
    state1_count_0fr = counts[1]
    error_fr0 = (state1_count_0fr - state0_count_0fr) / len(fr_states)
    
    unique, counts = np.unique(nfr_states, return_counts=True)
    state0_count_0nfr = counts[0]
    state1_count_0nfr = counts[1]
    error_nfr0 = (state1_count_0nfr - state0_count_0nfr) / len(fr_states)

    fig.add_trace(
        go.Heatmap(
            z=[[state0_count_0fr, state0_count_1fr],
                [state1_count_0fr, state1_count_1fr]],
            texttemplate= "%{z}",
            # texttemplate="%{text}",
            # textfont={"size":20}
            ),
    )
    
    fig.add_trace(
        go.Heatmap(
            z=[[state0_count_0nfr, state0_count_1nfr],
                [state1_count_0nfr, state1_count_1nfr]],
            texttemplate= "%{z}",
            # texttemplate="%{text}",
            # textfont={"size":20}
            ),
    )
    
    # fig.add_trace(
    #     go.Scatter(
    #         x=iterations,
    #         y=fr_states,
    #         mode="markers",
    #         name="fast reset",
    #         showlegend=True,
    #         legendgroup="group0",
    #     ),
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=iterations,
    #         y=nfr_states,
    #         mode="markers",
    #         name="no fast reset",
    #         showlegend=True,
    #         legendgroup="group1",
    #     ),
    # )

    # fitting_report += f"q{qubit}/r | state0 count: {state0_count:.0f}<br>"
    # fitting_report += f"q{qubit}/r | state1 count: {state1_count:.0f}<br>"
    # fitting_report += f"q{qubit}/r | Error FR: {error_fr:.6f}<br>"
    # fitting_report += f"q{qubit}/r | Fidelity FR(add 0 to 1 error): {(1 - error_fr):.6f}<br>"
    # fitting_report += f"q{qubit}/r | Error NFR: {error_nfr:.6f}<br>"
    # fitting_report += f"q{qubit}/r | Fidelity NFR(add 0 to 1 error): {(1 - error_nfr):.6f}<br>"

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
