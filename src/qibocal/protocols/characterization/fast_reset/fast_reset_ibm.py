from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

# IBM Fast Reset until saturation loop experiment
# https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/reset/backend_reset


@dataclass
class FastResetParameters(Parameters):
    """FastReset runcard inputs."""

    n_resets: Optional[int] = 1
    """Number of state resets."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class FastResetResults(Results):
    """FastReset outputs."""


FastResetType = np.dtype(
    [
        ("probability", np.float64),
    ]
)
"""Custom dtype for FastReset."""


@dataclass
class FastResetData(Data):
    """FastReset acquisition outputs."""

    n_resets: int
    data: dict[tuple[QubitId, int, bool], npt.NDArray[FastResetType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, probability, resets):
        """Store output for single qubit."""
        ar = np.empty(probability.shape, dtype=FastResetType)
        ar["probability"] = probability
        self.data[qubit, resets] = np.rec.array(ar)


def _acquisition(
    params: FastResetParameters, platform: Platform, qubits: Qubits
) -> FastResetData:
    """Data acquisition for resonator spectroscopy."""

    data = FastResetData(params.n_resets)

    for resets in np.arange(params.n_resets):
        # Define the pulse sequences
        RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            RX_pulses[qubit] = platform.create_RX90_pulse(qubit, start=0)
            sequence.add(RX_pulses[qubit])
            RX_pulses[qubit] = platform.reset_state(
                qubit, start=RX_pulses[qubit].finish, resets=resets
            )
            sequence.add(RX_pulses[qubit])
            RX_pulses[qubit] = platform.create_RX_pulse(
                qubit, start=RX_pulses[qubit].finish
            )
            sequence.add(RX_pulses[qubit])
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                qubit, start=RX_pulses[qubit].finish
            )
            sequence.add(ro_pulses[qubit])

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
            ),
        )

        # Save the data
        for ro_pulse in ro_pulses.values():
            result = results[ro_pulse.serial]
            print(result.probability(0))
            qubit = ro_pulse.qubit
            data.register_qubit(
                qubit,
                probability=result.probability(0),
                resets=resets,
            )

    return data


def _fit(data: FastResetData) -> FastResetResults:
    """Post-processing function for FastReset."""

    return FastResetResults()


def _plot(data: FastResetData, fit: FastResetResults, qubit):
    """Plotting function for FastReset."""

    # Maybe the plot can just be something like a confusion matrix between 0s and 1s ???
    figures = []
    fitting_report = ""

    data_plot = []
    for i in range(data.n_resets):
        y = [data[qubit, i].probability, 1 - data[qubit, i].probability]
        data_plot.append(
            go.Bar(
                x=[0, 1],
                name=str(i),
                y=y,
            )
        )

    fig = go.Figure(data=data_plot)
    figures.append(fig)

    # fitting_report +=
    # last part
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="State",
        yaxis_title="Probability",
    )

    return figures, fitting_report


fast_reset_ibm = Routine(_acquisition, _fit, _plot)
"""FastReset Routine object."""
