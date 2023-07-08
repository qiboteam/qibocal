from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import cumulative

MESH_SIZE = 50


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64)])
"""Custom dtype for rabi amplitude."""


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    data: dict[tuple[QubitId, int], npt.NDArray[ClassificationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, state, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=ClassificationType)
        ar["i"] = i
        ar["q"] = q
        self.data[qubit, state] = np.rec.array(ar)


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""

    threshold: dict[QubitId, float] = field(metadata=dict(update="threshold"))
    """Threshold for classification."""
    rotation_angle: dict[QubitId, float] = field(metadata=dict(update="iq_angle"))
    """Threshold for classification."""
    mean_gnd_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_gnd_states")
    )
    mean_exc_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_exc_states")
    )
    fidelity: dict[QubitId, float]
    assignment_fidelity: dict[QubitId, float]


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: Platform,
    qubits: Qubits,
) -> SingleShotClassificationData:
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.Platform`): custom abstract platform on which we perform the calibration.
        qubits (dict): dict of target Qubit objects to perform the action
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
        state0_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state0_results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit, state=0, i=result.voltage_i, q=result.voltage_q
        )

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state1_results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit, state=1, i=result.voltage_i, q=result.voltage_q
        )

    return data


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits
    thresholds, rotation_angles = {}, {}
    fidelities, assignment_fidelities = {}, {}
    mean_gnd_states = {}
    mean_exc_states = {}
    for qubit in qubits:
        iq_state0 = data[qubit, 0].i + 1.0j * data[qubit, 0].q
        iq_state1 = data[qubit, 1].i + 1.0j * data[qubit, 1].q

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

        real_values_combined = np.unique(
            np.concatenate((real_values_state1, real_values_state0))
        )
        real_values_combined.sort()

        cum_distribution_state1 = cumulative(real_values_combined, real_values_state1)
        cum_distribution_state0 = cumulative(real_values_combined, real_values_state0)
        cum_distribution_diff = np.abs(
            np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
        )
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 1)
        plt.scatter(
            np.sort(real_values_combined),
            cum_distribution_state1,
        )
        plt.scatter(
            np.sort(real_values_combined),
            cum_distribution_state0,
        )
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.scatter(
            np.sort(real_values_combined),
            cum_distribution_diff,
        )

        plt.legend()

        plt.subplot(1, 3, 3)
        plt.scatter(iq_state0_rotated.real, iq_state0_rotated.imag)
        plt.scatter(iq_state1_rotated.real, iq_state1_rotated.imag)
        plt.savefig("CUMULATIVE_CLS.png")

        argmax = np.argmax(cum_distribution_diff)
        threshold = real_values_combined[argmax]
        errors_state1 = data.nshots - cum_distribution_state1[argmax]
        errors_state0 = cum_distribution_state0[argmax]
        fidelity = cum_distribution_diff[argmax] / data.nshots
        assignment_fidelity = (errors_state1 + errors_state0) / data.nshots / 2
        thresholds[qubit] = threshold
        rotation_angles[
            qubit
        ] = -rotation_angle  # TODO: qblox driver np.rad2deg(-rotation_angle)
        fidelities[qubit] = fidelity
        mean_gnd_states[qubit] = [iq_mean_state0.real, iq_mean_state0.imag]
        mean_exc_states[qubit] = [iq_mean_state1.real, iq_mean_state1.imag]
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

    state0_data = data[qubit, 0]
    state1_data = data[qubit, 1]

    fig.add_trace(
        go.Scatter(
            x=state0_data.i,
            y=state0_data.q,
            name="Ground State",
            legendgroup="Ground State",
            mode="markers",
            showlegend=True,
            opacity=0.7,
            marker=dict(size=3),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=state1_data.i,
            y=state1_data.q,
            name="Excited State",
            legendgroup="Excited State",
            mode="markers",
            showlegend=True,
            opacity=0.7,
            marker=dict(size=3),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[np.mean(state0_data.i)],
            y=[np.mean(state0_data.q)],
            name="Average Ground State",
            legendgroup="Average Ground State",
            showlegend=True,
            mode="markers",
            marker=dict(size=10),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[np.mean(state1_data.i)],
            y=[np.mean(state1_data.q)],
            name="Average Excited State",
            legendgroup="Average Excited State",
            showlegend=True,
            mode="markers",
            marker=dict(size=10),
        ),
    )

    max_x = max(
        max_x,
        np.max(state0_data.i),
        np.max(state1_data.i),
    )
    max_y = max(
        max_y,
        np.max(state0_data.q),
        np.max(state1_data.q),
    )
    min_x = min(
        min_x,
        np.min(state0_data.i),
        np.min(state1_data.i),
    )
    min_y = min(
        min_y,
        np.min(state0_data.q),
        np.min(state1_data.q),
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
            opacity=0.4,
            name="Score",
            hoverinfo="skip",
        ),
    )

    fitting_report = (
        fitting_report
        + f"{qubit} | Average Ground State (i,q): ({fit.mean_gnd_states[qubit][0]:.3f}, {fit.mean_gnd_states[qubit][1]:.3f}) <br>"
        + f"{qubit} | Average Excited State (i,q): ({fit.mean_exc_states[qubit][0]:.3f}, {fit.mean_exc_states[qubit][1]:.3f}) <br>"
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
