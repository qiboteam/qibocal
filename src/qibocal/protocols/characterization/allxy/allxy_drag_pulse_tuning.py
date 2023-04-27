from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from ....auto.operation import Parameters, Qubits, Results, Routine
from ....data import Data
from ....plots.utils import get_color
from .allxy import add_gate_pair_pulses_to_sequence, gatelist


@dataclass
class AllXYDragParameters(Parameters):
    beta_start: float
    beta_end: float
    beta_step: float

@dataclass
class AllXYDragResults(Results):
    ...


class AllXYDragData(Data):
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
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape
    in order to find the optimal drag pulse coefficient for pi pulses.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        beta_start (float): Initial drag pulse beta parameter
        beta_end (float): Maximum drag pulse beta parameter
        beta_step (float): Scan range step for the drag pulse beta parameter
        software_averages (int): Number of executions of the routine for averaging results

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Difference between resonator signal voltage mesurement in volts from sequence 1 and 2
            - **i[V]**: Difference between resonator signal voltage mesurement for the component I in volts from sequence 1 and 2
            - **q[V]**: Difference between resonator signal voltage mesurement for the component Q in volts from sequence 1 and 2
            - **phase[rad]**: Difference between resonator signal phase mesurement in radians from sequence 1 and 2
            - **probability[dimensionless]**: Probability of being in |0> state
            - **gateNumber[dimensionless]**: Gate number applied from the list of gates
            - **beta_param[dimensionless]**: Beta paramter applied in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    data = AllXYDragData()

    # repeat the experiment as many times as defined by software_averages
    count = 0
    # sweep the parameters
    for beta_param in np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4):
        gateNumber = 1
        for gates in gatelist:
            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                sequence, ro_pulses[qubit] = add_gate_pair_pulses_to_sequence(
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
    # finally, save the remaining data
    return data


def _fit(_data: AllXYDragData) -> AllXYDragResults:
    return AllXYDragResults()


def _plot(data: AllXYDragData, _fit: AllXYDragResults, qubit):
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
                text=gatelist,
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
