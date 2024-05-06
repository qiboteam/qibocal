from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from . import allxy


@dataclass
class AllXYDragParameters(Parameters):
    """AllXYDrag runcard inputs."""

    delay_start: float
    """Initial delay parameter for resonator depletion."""
    delay_end: float
    """Final delay parameter for resonator depletion."""
    delay_step: float
    """Step delay parameter for resonator depletion."""


@dataclass
class AllXYDragResults(Results):
    """AllXYDrag outputs."""


@dataclass
class AllXYDragData(Data):
    """AllXY acquisition outputs."""

    delay_param: Optional[float] = None
    """Delay parameter for resonator depletion."""
    data: dict[tuple[QubitId, float], npt.NDArray[allxy.AllXYType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def delay_params(self):
        """Access qubits from data structure."""
        return np.unique([b[1] for b in self.data])


def _acquisition(
    params: AllXYDragParameters,
    platform: Platform,
    targets: list[QubitId],
) -> AllXYDragData:
    r"""
    Data acquisition for allXY experiment varying delay after a measurement.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape
    in order to find the optimal drag pulse coefficient for pi pulses.
    """

    data = AllXYDragData()

    delays = np.arange(params.delay_start, params.delay_end, params.delay_step).round(4)
    # sweep the parameters
    for delay_param in delays:
        for gates in allxy.gatelist:
            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in targets:
                ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
                sequence.add(ro_pulse)
                sequence, ro_pulses[qubit] = allxy.add_gate_pair_pulses_to_sequence(
                    platform,
                    gates,
                    qubit,
                    sequence,
                    sequence_delay=int(
                        delay_param
                    ),  # We need conversion to int due to devices for now
                    readout_delay=996,  # because we already add a +4
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
            for qubit in targets:
                z_proj = 2 * results[ro_pulses[qubit].serial].probability(0) - 1
                # store the results
                gate = "-".join(gates)
                data.register_qubit(
                    allxy.AllXYType,
                    (qubit, float(delay_param)),
                    dict(prob=np.array([z_proj]), gate=np.array([gate])),
                )
    return data


def _fit(_data: AllXYDragData) -> AllXYDragResults:
    """Post-processing for allXYDrag."""
    return AllXYDragResults()


def _plot(data: AllXYDragData, target: QubitId, fit: AllXYDragResults = None):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = ""

    fig = go.Figure()
    delay_params = data.delay_params

    for j, delay_param in enumerate(delay_params):
        delay_param_data = data[target, delay_param]
        fig.add_trace(
            go.Scatter(
                x=delay_param_data.gate,
                y=delay_param_data.prob,
                mode="markers+lines",
                opacity=0.5,
                name=f"Delay {delay_param}",
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
        xaxis_title="Gate sequence number",
        yaxis_title="Expectation value of Z",
    )

    figures.append(fig)

    return figures, fitting_report


allxy_resonator_depletion_tuning = Routine(_acquisition, _fit, _plot)
"""AllXYDrag Routine object."""
