from dataclasses import dataclass
from typing import List

import numpy as np
import plotly.graph_objects as go
from qibo import gates, models, set_backend

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class RotationParameters(Parameters):
    nshots: int
    theta_start: float
    theta_end: float
    samples: float


@dataclass
class RotationData(Data):
    thetas: List[float]
    probabilities: List[float]
    simulated_probabilities: List[float]

    def save(self, path):
        np.savetxt(
            path / "test.txt",
            np.array([self.thetas, self.probabilities, self.simulated_probabilities]),
        )


@dataclass
class RotationResults(Results):
    """"""


def _acquisition(params: RotationParameters) -> RotationData:
    thetas = np.linspace(params.theta_start, params.theta_end, params.samples)
    probabilities = []
    simulated_probabilities = []

    for theta in thetas:
        circuit = models.Circuit(1)
        circuit.add(gates.RX(0, theta))
        circuit.add(gates.M(0))
        result = circuit(nshots=params.nshots)
        prob = result.probabilities()[0]
        probabilities.append(prob)

    set_backend("numpy")

    for theta in thetas:
        circuit = models.Circuit(1)
        circuit.add(gates.RX(0, theta))
        circuit.add(gates.M(0))
        result = circuit(nshots=params.nshots)
        prob = result.probabilities()[0]
        simulated_probabilities.append(prob)

    return RotationData(thetas, probabilities, simulated_probabilities)


def _plot(data):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data.thetas,
            y=data.simulated_probabilities,
            name="Simulation",
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=data.thetas,
            y=data.probabilities,
            name="QPU",
            mode="markers",
        ),
    )

    figure.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title=r"Rotation angle (rad)",
        yaxis_title="Ground State Probability",
    )

    return [figure], ""


rotation = Routine(_acquisition, None, _plot)
