from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine


@dataclass
class AllXYParameters(Parameters):
    """AllXY runcard inputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class AllXYResults(Results):
    """AllXY outputs."""


AllXYType = np.dtype([("prob", np.float64), ("gate", np.int64)])


@dataclass
class AllXYData(Data):
    """AllXY acquisition outputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""
    data: Dict[QubitId, npt.NDArray[AllXYType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, prob, gate):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=AllXYType)
        ar["prob"] = prob
        ar["gate"] = gate
        if self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)

    @property
    def qubits(self):
        """Access qubits from data structure."""
        return [q for q in self.data]

    def __getitem__(self, qubit):
        return self.data[qubit]

    def save(self, path):
        """Store results."""
        self.to_npz(path, self.data)


gatelist = [
    ["I", "I"],
    ["RX(pi)", "RX(pi)"],
    ["RY(pi)", "RY(pi)"],
    ["RX(pi)", "RY(pi)"],
    ["RY(pi)", "RX(pi)"],
    ["RX(pi/2)", "I"],
    ["RY(pi/2)", "I"],
    ["RX(pi/2)", "RY(pi/2)"],
    ["RY(pi/2)", "RX(pi/2)"],
    ["RX(pi/2)", "RY(pi)"],
    ["RY(pi/2)", "RX(pi)"],
    ["RX(pi)", "RY(pi/2)"],
    ["RY(pi)", "RX(pi/2)"],
    ["RX(pi/2)", "RX(pi)"],
    ["RX(pi)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi)"],
    ["RY(pi)", "RY(pi/2)"],
    ["RX(pi)", "I"],
    ["RY(pi)", "I"],
    ["RX(pi/2)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi/2)"],
]


def _acquisition(
    params: AllXYParameters,
    platform: Platform,
    qubits: Qubits,
) -> AllXYData:
    r"""
    Data acquisition for allXY experiment.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.
    """

    # create a Data object to store the results
    data = AllXYData(params.beta_param)

    # repeat the experiment as many times as defined by software_averages
    # for iteration in range(params.software_averages):
    for gateNumber, gates in enumerate(gatelist):
        # create a sequence of pulses
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            sequence, ro_pulses[qubit] = add_gate_pair_pulses_to_sequence(
                platform, gates, qubit, sequence, params.beta_param
            )

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        # retrieve the results for every qubit
        for qubit in qubits:
            z_proj = 2 * results[ro_pulses[qubit].serial].probability(0) - 1
            # store the results
            data.register_qubit(qubit, z_proj, gateNumber)
    # finally, save the remaining data
    return data


def add_gate_pair_pulses_to_sequence(
    platform: Platform,
    gates,
    qubit,
    sequence,
    beta_param=None,
):
    pulse_duration = platform.create_RX_pulse(qubit, start=0).duration
    # All gates have equal pulse duration

    sequenceDuration = 0
    pulse_start = 0

    for gate in gates:
        if gate == "I":
            # print("Transforming to sequence I gate")
            pass

        if gate == "RX(pi)":
            # print("Transforming to sequence RX(pi) gate")
            if beta_param == None:
                RX_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX_pulse)

        if gate == "RX(pi/2)":
            # print("Transforming to sequence RX(pi/2) gate")
            if beta_param == None:
                RX90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX90_pulse)

        if gate == "RY(pi)":
            # print("Transforming to sequence RY(pi) gate")
            if beta_param == None:
                RY_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY_pulse)

        if gate == "RY(pi/2)":
            # print("Transforming to sequence RY(pi/2) gate")
            if beta_param == None:
                RY90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY90_pulse)

        sequenceDuration = sequenceDuration + pulse_duration
        pulse_start = pulse_duration

    # RO pulse starting just after pair of gates
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=sequenceDuration + 4)
    sequence.add(ro_pulse)
    return sequence, ro_pulse


def _fit(_data: AllXYData) -> AllXYResults:
    """Post-Processing for allXY"""
    return AllXYResults()


# allXY
def _plot(data: AllXYData, _fit: AllXYResults, qubit):
    """Plotting function for allXY."""

    figures = []
    fitting_report = "No fitting data"
    fig = go.Figure()

    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.gate,
            y=qubit_data.prob,
            mode="markers",
            text=gatelist,
            textposition="bottom center",
            name="Expectation value",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )
    fig.add_hline(
        y=1,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )

    fig.add_hline(
        y=-1,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequence number",
        yaxis_title="Expectation value of Z",
    )

    figures.append(fig)

    return figures, fitting_report


allxy = Routine(_acquisition, _fit, _plot)
"""AllXY Routine object."""
