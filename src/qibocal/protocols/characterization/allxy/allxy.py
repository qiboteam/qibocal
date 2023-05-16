from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import Data
from qibocal.plots.utils import get_color


@dataclass
class AllXYParameters(Parameters):
    """AllXY runcard inputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""


@dataclass
class AllXYResults(Results):
    """AllXY outputs."""


class AllXYData(Data):
    """AllXY acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"probability", "gateNumber", "qubit"},
        )


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
    platform: AbstractPlatform,
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
    data = AllXYData()

    # repeat the experiment as many times as defined by software_averages
    # for iteration in range(params.software_averages):
    gateNumber = 1
    # sweep the parameter
    for gateNumber, gates in enumerate(gatelist):
        # create a sequence of pulses
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            sequence, ro_pulses[qubit] = add_gate_pair_pulses_to_sequence(
                platform, gates, qubit, sequence, params.beta_param
            )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                z_proj = 2 * results[ro_pulse.serial].ground_state_probability - 1
                # store the results
                r = {
                    "probability": z_proj,
                    "gateNumber": gateNumber,
                    "beta_param": params.beta_param,
                    "qubit": ro_pulse.qubit,
                }
                data.add(r)
    # finally, save the remaining data
    return data


def add_gate_pair_pulses_to_sequence(
    platform: AbstractPlatform,
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

    qubit_data = data.df[data.df["qubit"] == qubit].drop(columns=["qubit"])

    fig.add_trace(
        go.Scatter(
            x=qubit_data["gateNumber"],
            y=qubit_data["probability"],
            marker_color=get_color(0),
            mode="markers",
            text=gatelist,
            textposition="bottom center",
            opacity=0.3,
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
