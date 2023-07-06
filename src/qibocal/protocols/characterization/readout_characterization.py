from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

@dataclass
class ReadoutCharacterizationParameters(Parameters):
    """ReadoutCharacterization runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ReadoutCharacterizationResults(Results):
    """ReadoutCharacterization outputs."""

    optimal_integration_weights: Dict[QubitId, float]
    """
    Optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.
    """


ReadoutCharacterizationType = np.dtype(
    [
        ("probability", np.float64),
    ]
)
"""Custom dtype for ReadoutCharacterization."""


@dataclass
class ReadoutCharacterizationData(Data):
    """ReadoutCharacterization acquisition outputs."""

    data: dict[tuple[QubitId, int, bool], npt.NDArray[ReadoutCharacterizationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, probability, state, number_readout):
        """Store output for single qubit."""
        ar = np.empty(probability.shape, dtype=ReadoutCharacterizationType)
        ar["probability"] = probability
        self.data[qubit, state, number_readout] = np.rec.array(ar)


def _acquisition(
    params: ReadoutCharacterizationParameters, platform: Platform, qubits: Qubits
) -> ReadoutCharacterizationData:
    """Data acquisition for resonator spectroscopy."""

    data = ReadoutCharacterizationData()
    
    
    #FIXME: ADD 1st measurament and post_selection for accurate state preparation ?
    
    
    for state in [0, 1]:
        # Define the pulse sequences
        if state == 1:
            RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            if state == 1:
                RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                start = RX_pulses[qubit].finish
                ro_pulses[qubit] = []
                for _ in range(2):
                    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
                    start += ro_pulse.duration
                    sequence.add(ro_pulse)
                    ro_pulses[qubit].append(ro_pulse)
                
                sequence.add(RX_pulses[qubit])
            else:
                start = 0
                ro_pulses[qubit] = []
                for _ in range(2):
                    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start)
                    start += ro_pulse.duration
                    sequence.add(ro_pulse)
                    ro_pulses[qubit].append(ro_pulse)

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
            ),
        )

        # Save the data
        for qubit in qubits:
            i = 0
            for ro_pulse in ro_pulses[qubit]:
                result = results[ro_pulse.serial]
                qubit = ro_pulse.qubit
                data.register_qubit(
                    qubit,
                    probability=result.samples,
                    state=state,
                    number_readout = i,
                )

    return data


def _fit(data: ReadoutCharacterizationData) -> ReadoutCharacterizationResults:
    """Post-processing function for ReadoutCharacterization."""

    # Getting some kind of fidelity

    return ReadoutCharacterizationResults({})


def _plot(data: ReadoutCharacterizationData, fit: ReadoutCharacterizationResults, qubit):
    """Plotting function for ReadoutCharacterization."""

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

    
    
    
    # 1st measurement (m=1)
    m1_states = data[qubit, 1, 0].probability
    nshots = len(m1_states)
    # state 1
    unique, counts = np.unique(m1_states, return_counts=True)
    state0_count_1_m1 = counts[0]
    state1_count_1_m1 = counts[1]
    # state 0
    unique, counts = np.unique(_m1_states, return_counts=True)
    state0_count_0_m1 = counts[0]
    state1_count_0_m1 = counts[1]
    
    Lambda_M =[
                [state1_count_0_m1, state1_count_1_m1],
                [state0_count_0_m1, state0_count_1_m1],
            ]
    
    fidelity = 1 - (state0_count_0_m1 + state1_count_1_m1) / 2
    
    
    P_0o_1i = 
    P_0o_1i = 
    
    # state 1
    fr_states = data[qubit, 1, True].probability
    
    
    # FIXME crashes if all states are on the same counts
    unique, counts = np.unique(fr_states, return_counts=True)
    state0_count_1fr = counts[0]
    state1_count_1fr = counts[1]
    error_fr1 = 1 - (nshots - state0_count_1fr) / nshots

    

    # state 0
    fr_states = data[qubit, 0, True].probability
    nfr_states = data[qubit, 0, False].probability

    unique, counts = np.unique(fr_states, return_counts=True)
    state0_count_0fr = counts[0]
    state1_count_0fr = counts[1]
    error_fr0 = (nshots - state0_count_0fr) / nshots

    

    fig.add_trace(
        go.Heatmap(
            z=[
                [state1_count_0fr, state1_count_1fr],
                [state0_count_0fr, state0_count_1fr],
            ],
            
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

    fig.add_trace(
        go.Heatmap(
            z=[
                [state1_count_0nfr, state1_count_1nfr],
                [state0_count_0nfr, state0_count_1nfr],
            ],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text=f"{qubit}: Relaxation Time",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Shot", row=1, col=2)

    fitting_report += f"q{qubit}/r | Error FR0: {error_fr0:.6f}<br>"
    fitting_report += f"q{qubit}/r | Error FR1: {error_fr1:.6f}<br>"
    fitting_report += f"q{qubit}/r | Assigment Fidelity FR: {(1 - (error_fr0 + error_fr1)/2):.6f}<br>"
    fitting_report += f"q{qubit}/r | Error NFR0: {error_nfr0:.6f}<br>"
    fitting_report += f"q{qubit}/r | Error NFR1: {error_nfr1:.6f}<br>"
    fitting_report += (
        f"q{qubit}/r | Assigment Fidelity NFR: {(1 - (error_nfr0 + error_nfr1)/2):.6f}<br>"
    )

    # last part
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="State prepared",
        yaxis_title="State read",
    )

    figures.append(fig)

    return figures, fitting_report


readout_characterization = Routine(_acquisition, _fit, _plot)
"""ReadoutCharacterization Routine object."""
