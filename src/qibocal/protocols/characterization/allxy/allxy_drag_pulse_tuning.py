from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import Data
from qibocal.plots.utils import get_color

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


@dataclass
class AllXYDragResults(Results):
    """AllXYDrag outputs."""


class AllXYDragData(Data):
    """AllXYDrag acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={
                "probability",
                "gateNumber",
                "beta_param",
                "qubit",
            },
        )


def _acquisition(
    params: AllXYDragParameters,
    platform: AbstractPlatform,
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

    count = 0
    # sweep the parameters
    for beta_param in np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4):
        gateNumber = 1
        for gates in allxy.gatelist:
            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                sequence, ro_pulses[qubit] = allxy.add_gate_pair_pulses_to_sequence(
                    platform, gates, qubit, sequence, beta_param
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
                    "beta_param": beta_param,
                    "qubit": ro_pulse.qubit,
                }
                data.add(r)
            gateNumber += 1
    return data


def _fit(_data: AllXYDragData) -> AllXYDragResults:
    """Post-processing for allXYDrag."""
    return AllXYDragResults()


def _plot(data: AllXYDragData, _fit: AllXYDragResults, qubit):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = "No fitting data"

    fig = go.Figure()
    beta_params = data.df["beta_param"].unique()

    qubit_data = data.df[data.df["qubit"] == qubit]

    for j, beta_param in enumerate(beta_params):
        beta_param_data = qubit_data[qubit_data["beta_param"] == beta_param]
        fig.add_trace(
            go.Scatter(
                x=beta_param_data["gateNumber"],
                y=beta_param_data["probability"],
                marker_color=get_color(j),
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
