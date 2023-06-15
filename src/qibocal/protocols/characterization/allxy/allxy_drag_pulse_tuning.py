from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from . import allxy


@dataclass
class AllXYDragParameters(Parameters):
    """AllXYDrag runcard inputs."""

    beta_start: float
    """Initial beta parameter for Drag pulse."""
    beta_end: float
    """Final beta parameter for Drag pulse."""
    beta_step: float
    """Step beta parameter for Drag pulse."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class AllXYDragResults(Results):
    """AllXYDrag outputs."""


class AllXYDragData(Data):
    """AllXYDrag acquisition outputs."""


@dataclass
class AllXYDragData(Data):
    """AllXY acquisition outputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""
    data: Dict[Tuple[QubitId, int], npt.NDArray[allxy.AllXYType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, beta, prob, gate):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=allxy.AllXYType)
        ar["prob"] = prob
        ar["gate"] = gate
        if (qubit, beta) in self.data:
            self.data[qubit, beta] = np.rec.array(
                np.concatenate((self.data[qubit, beta], ar))
            )
        else:
            self.data[qubit, beta] = np.rec.array(ar)

    @property
    def qubits(self):
        """Access qubits from data structure."""
        return np.unique([q[0] for q in self.data])

    @property
    def beta_params(self):
        """Access qubits from data structure."""
        return np.unique([b[1] for b in self.data])

    def __getitem__(self, qubit_beta: tuple):
        qubit, beta = qubit_beta
        return self.data[qubit, beta]

    def save(self, path):
        """Store results."""
        self.to_npz(path, self.data)


def _acquisition(
    params: AllXYDragParameters,
    platform: Platform,
    qubits: Qubits,
) -> AllXYDragData:
    r"""
    Data acquisition for allXY experiment varying beta.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape
    in order to find the optimal drag pulse coefficient for pi pulses.
    """

    data = AllXYDragData()

    # sweep the parameters
    for beta_param in np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4):
        for gateNumber, gates in enumerate(allxy.gatelist):
            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                sequence, ro_pulses[qubit] = allxy.add_gate_pair_pulses_to_sequence(
                    platform, gates, qubit, sequence, beta_param
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )

            # retrieve the results for every qubit
            for qubit in qubits:
                z_proj = 2 * results[ro_pulses[qubit].serial].probability(0) - 1
                # store the results
                data.register_qubit(qubit, beta_param, z_proj, gateNumber)
    return data


def _fit(_data: AllXYDragData) -> AllXYDragResults:
    """Post-processing for allXYDrag."""
    return AllXYDragResults()


def _plot(data: AllXYDragData, _fit: AllXYDragResults, qubit):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = "No fitting data"

    fig = go.Figure()
    beta_params = data.beta_params

    for j, beta_param in enumerate(beta_params):
        beta_param_data = data[qubit, beta_param]
        fig.add_trace(
            go.Scatter(
                x=beta_param_data.gate,
                y=beta_param_data.prob,
                mode="markers+lines",
                opacity=0.5,
                name=f"Beta {beta_param}",
                showlegend=True,
                legendgroup=f"group{j}",
                text=allxy.gatelist,
                textposition="bottom center",
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


allxy_drag_pulse_tuning = Routine(_acquisition, _fit, _plot)
"""AllXYDrag Routine object."""
