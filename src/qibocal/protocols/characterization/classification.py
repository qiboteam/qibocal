from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color_state0, get_color_state1

MESH_SIZE = 50


@dataclass
class SingleShotClassificationParameters(Parameters):
    nshots: int


class SingleShotClassificationData(DataUnits):
    def __init__(self, nshots):
        super().__init__(
            "data",
            options=["qubit", "state"],
        )

        self._nshots = nshots

    @property
    def nshots(self):
        return self._nshots


@dataclass
class SingleShotClassificationResults(Results):
    threshold: Dict[List[Tuple], str] = field(metadata=dict(update="threshold"))
    rotation_angle: Dict[List[Tuple], str] = field(metadata=dict(update="iq_angle"))
    mean_gnd_states: Dict[List[Tuple], str] = field(
        metadata=dict(update="mean_gnd_states")
    )
    mean_exc_states: Dict[List[Tuple], str] = field(
        metadata=dict(update="mean_exc_states")
    )
    fidelity: Dict[List[Tuple], str]
    assignment_fidelity: Dict[List[Tuple], str]


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> SingleShotClassificationData:
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = SingleShotClassificationData(params.nshots)

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence, nshots=params.nshots
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state0_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * params.nshots,
                "state": [0] * params.nshots,
            }
        )
        data.add_data_from_dict(r)

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence, nshots=params.nshots
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state1_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * params.nshots,
                "state": [1] * params.nshots,
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.df["qubit"].unique()
    thresholds, rotation_angles = {}, {}
    fidelities, assignment_fidelities = {}, {}
    mean_gnd_states = {}
    mean_exc_states = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit].drop(
            columns=["qubit", "MSR", "phase"]
        )

        iq_state0 = (
            qubit_data[qubit_data["state"] == 0]["i"].pint.to("V").pint.magnitude
            + 1.0j
            * qubit_data[qubit_data["state"] == 0]["q"].pint.to("V").pint.magnitude
        )
        iq_state1 = (
            qubit_data[qubit_data["state"] == 1]["i"].pint.to("V").pint.magnitude
            + 1.0j
            * qubit_data[qubit_data["state"] == 1]["q"].pint.to("V").pint.magnitude
        )

        iq_state1 = np.array(iq_state1)
        iq_state0 = np.array(iq_state0)

        iq_mean_state1 = np.mean(iq_state1)
        iq_mean_state0 = np.mean(iq_state0)

        vector01 = iq_mean_state1 - iq_mean_state0
        rotation_angle = np.angle(vector01)

        iq_state1_rotated = iq_state1 * np.exp(-1j * rotation_angle)
        iq_state0_rotated = iq_state0 * np.exp(-1j * rotation_angle)

        real_values_state1 = iq_state1_rotated.real
        real_values_state0 = iq_state0_rotated.real

        real_values_combined = np.concatenate((real_values_state1, real_values_state0))
        real_values_combined.sort()

        cum_distribution_state1 = [
            sum(map(lambda x: x.real >= real_value, real_values_state1))
            for real_value in real_values_combined
        ]
        cum_distribution_state0 = [
            sum(map(lambda x: x.real >= real_value, real_values_state0))
            for real_value in real_values_combined
        ]

        cum_distribution_diff = np.abs(
            np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
        )
        argmax = np.argmax(cum_distribution_diff)
        threshold = real_values_combined[argmax]
        errors_state1 = data.nshots - cum_distribution_state1[argmax]
        errors_state0 = cum_distribution_state0[argmax]
        fidelity = cum_distribution_diff[argmax] / data.nshots
        assignment_fidelity = 1 - (errors_state1 + errors_state0) / data.nshots / 2
        thresholds[qubit] = threshold
        rotation_angles[
            qubit
        ] = -rotation_angle  # TODO: qblox driver np.rad2deg(-rotation_angle)
        fidelities[qubit] = fidelity
        mean_gnd_states[qubit] = iq_mean_state0
        mean_exc_states[qubit] = iq_mean_state1
        assignment_fidelities[qubit] = assignment_fidelity

    return SingleShotClassificationResults(
        thresholds,
        rotation_angles,
        mean_gnd_states,
        mean_exc_states,
        fidelities,
        assignment_fidelities,
    )


def _plot(
    data: SingleShotClassificationData, fit: SingleShotClassificationResults, qubit
):
    figures = []

    fig = go.Figure()

    fitting_report = ""
    max_x, max_y, min_x, min_y = 0, 0, 0, 0

    qubit_data = data.df[data.df["qubit"] == qubit]
    state0_data = qubit_data[data.df["state"] == 0].drop(
        columns=["MSR", "phase", "qubit"]
    )
    state1_data = qubit_data[data.df["state"] == 1].drop(
        columns=["MSR", "phase", "qubit"]
    )

    fig.add_trace(
        go.Scatter(
            x=state0_data["i"].pint.to("V").pint.magnitude,
            y=state0_data["q"].pint.to("V").pint.magnitude,
            name="Ground State",
            legendgroup="Ground State",
            mode="markers",
            showlegend=True,
            opacity=0.7,
            marker=dict(size=3, color=get_color_state0(0)),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=state1_data["i"].pint.to("V").pint.magnitude,
            y=state1_data["q"].pint.to("V").pint.magnitude,
            name="Excited State",
            legendgroup="Excited State",
            mode="markers",
            showlegend=True,
            opacity=0.7,
            marker=dict(size=3, color=get_color_state1(0)),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[state0_data["i"].pint.to("V").pint.magnitude.mean()],
            y=[state0_data["q"].pint.to("V").pint.magnitude.mean()],
            name="Average Ground State",
            legendgroup="Average Ground State",
            showlegend=True,
            mode="markers",
            marker=dict(size=10, color=get_color_state0(0)),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[state1_data["i"].pint.to("V").pint.magnitude.mean()],
            y=[state1_data["q"].pint.to("V").pint.magnitude.mean()],
            name="Average Excited State",
            legendgroup="Average Excited State",
            showlegend=True,
            mode="markers",
            marker=dict(size=10, color=get_color_state1(0)),
        ),
    )

    max_x = max(
        max_x,
        state0_data["i"].pint.to("V").pint.magnitude.max(),
        state1_data["i"].pint.to("V").pint.magnitude.max(),
    )
    max_y = max(
        max_y,
        state0_data["q"].pint.to("V").pint.magnitude.max(),
        state1_data["q"].pint.to("V").pint.magnitude.max(),
    )
    min_x = min(
        min_x,
        state0_data["i"].pint.to("V").pint.magnitude.min(),
        state1_data["i"].pint.to("V").pint.magnitude.min(),
    )
    min_y = min(
        min_y,
        state0_data["q"].pint.to("V").pint.magnitude.min(),
        state1_data["q"].pint.to("V").pint.magnitude.min(),
    )

    feature_x = np.linspace(min_x, max_x, MESH_SIZE)
    feature_y = np.linspace(min_y, max_y, MESH_SIZE)

    [x, y] = np.meshgrid(feature_x, feature_y)

    z = (
        (np.exp(1j * fit.rotation_angle[qubit]) * (x + 1j * y)).real
        > fit.threshold[qubit]
    ).astype(int)
    fig.add_trace(
        go.Contour(
            x=feature_x,
            y=feature_y,
            z=z,
            showscale=False,
            colorscale=[get_color_state0(0), get_color_state1(0)],
            opacity=0.4,
            name="Score",
            hoverinfo="skip",
        ),
    )

    fitting_report = (
        fitting_report
        + f"{qubit} | Average Ground State: {fit.mean_gnd_states[qubit]:.4f} <br>"
        + f"{qubit} | Average Excited State: {fit.mean_exc_states[qubit]:.4f} <br>"
        + f"{qubit} | Rotation Angle: {fit.rotation_angle[qubit]:.3f} rad <br>"
        + f"{qubit} | Threshold: {fit.threshold[qubit]:.4f} <br>"
        + f"{qubit} | Fidelity: {fit.fidelity[qubit]:.3f} <br>"
        + f"{qubit} | Assignment Fidelity: {fit.assignment_fidelity[qubit]:.3f} <br>"
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        xaxis_range=(min_x, max_x),
        yaxis_range=(min_y, max_y),
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    figures.append(fig)

    return figures, fitting_report


single_shot_classification = Routine(_acquisition, _fit, _plot)
