from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo import Circuit, gates
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from ...auto.operation import Data, Parameters, Qubits, Results, Routine


@dataclass
class RotationParameters(Parameters):
    """Parameters for rotation protocol."""

    theta_start: float
    """Initial angle."""
    theta_end: float
    """Final angle."""
    theta_step: float
    """Angle step."""
    nshots: int
    """Number of shots."""


RotationType = np.dtype([("theta", np.float64), ("prob", np.float64)])


@dataclass
class RotationData(Data):
    """Rotation data."""

    data: dict[QubitId, npt.NDArray[RotationType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, theta, prob):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=RotationType)
        ar["theta"] = theta
        ar["prob"] = prob
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


@dataclass
class RotationResults(Results):
    """Results object for data"""

    fitted_parameters: dict[QubitId, list] = field(default_factory=dict)


def acquisition(
    params: RotationParameters,
    platform: Platform,
    qubits: Qubits,
) -> RotationData:
    r"""
    Data acquisition for rotation routine.

    Args:
        params (:class:`RotationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`RotationData`)
    """

    angles = np.linspace(params.theta_start, params.theta_end, params.theta_step)
    # create a data structure
    data = RotationData()
    # sweep the parameter
    for angle in angles:
        # create a sequence of pulses for the experiment
        circuit = Circuit(platform.nqubits)
        for qubit in qubits:
            circuit.add(gates.RX(qubit, theta=angle))
            circuit.add(gates.M(qubit))

        result = circuit(nshots=params.nshots)
        # print(result[qubit])
        for qubit in qubits:
            prob = result.probabilities(qubits=[qubit])[0]
            data.register_qubit(qubit, theta=angle, prob=prob)

    return data


def sin_fit(x, offset, amplitude, omega):
    return offset + amplitude * np.sin(omega * x)


def fit(data: RotationData) -> RotationResults:
    qubits = data.qubits
    freqs = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        thetas = qubit_data.theta
        probs = qubit_data.prob

        popt, _ = curve_fit(sin_fit, thetas, probs)

        freqs[qubit] = popt[2] / 2 * np.pi
        fitted_parameters[qubit] = popt.tolist()

    return RotationResults(
        fitted_parameters=fitted_parameters,
    )


def plot(data: RotationData, fit: RotationResults, qubit):
    """Plotting function for rotation."""

    figures = []
    fig = go.Figure()

    fitting_report = "No fitting data"
    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.theta,
            y=qubit_data.prob,
            opacity=1,
            name="Probability",
            showlegend=True,
            legendgroup="Voltage",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=qubit_data.theta,
            y=sin_fit(
                qubit_data.theta,
                *fit.fitted_parameters[qubit],
            ),
            name="Fit",
            line=go.scatter.Line(dash="dot"),
        ),
    )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Theta [rad]",
        yaxis_title="Probability",
    )

    figures.append(fig)

    return figures, fitting_report


rotation = Routine(acquisition, fit, plot)
"""Rotation Routine  object."""
