from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class StateTomographyParameters(Parameters):
    """Tomography input parameters"""

    circuit: Optional[Circuit] = None
    """Circuit to prepare initial state"""


TomographyType = np.dtype(
    [
        ("sim", np.int64),
        ("hardware", np.int64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class StateTomographyData(Data):
    """Tomography data"""

    data: dict[tuple[str, QubitId], npt.NDArray[TomographyType]] = field(
        default_factory=dict
    )


@dataclass
class StateTomographyResults(Results):
    """Tomography results"""


def _acquisition(
    params: StateTomographyParameters, platform: Platform, targets: list[QubitId]
):
    """Acquisition protocol for state tomography."""

    simulation = {}
    hardware = {}
    if params.circuit is None:
        params.circuit = Circuit(
            platform.nqubits, wire_names=[str(i) for i in range(platform.nqubits)]
        )
        # params.circuit.add(gates.M(i) for i in targets)

    data = StateTomographyData()

    for basis in ["X", "Y", "Z"]:
        basis_circuit = deepcopy(params.circuit)
        if basis != "Z":
            basis_circuit.add(getattr(gates, basis)(0).basis_rotation())

        # basis_circuit.add(gates.M(i, basis=getattr(gates, basis)) for i in targets)

        basis_circuit.add(gates.M(0))
        data.register_qubit(
            TomographyType,
            (basis, 0),
            dict(
                sim=NumpyBackend()
                .execute_circuit(basis_circuit, nshots=params.nshots)
                .samples(),
                hardware=basis_circuit(nshots=params.nshots).samples(),
            ),
        )

    return data


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""

    return StateTomographyResults()


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitId):
    """Plotting for state tomography"""
    fig = go.Figure()
    simulation = [2 * np.mean(data[basis, 0].sim) - 1 for basis in ["X", "Y", "Z"]]
    hardware = [2 * np.mean(data[basis, 0].hardware) - 1 for basis in ["X", "Y", "Z"]]
    fig.add_trace(go.Bar(x=["X", "Y", "Z"], y=simulation, name="Simulation"))
    fig.add_trace(go.Bar(x=["X", "Y", "Z"], y=hardware, name="Hardware"))
    fig.update_layout(barmode="group")
    return [fig], ""


state_tomography = Routine(_acquisition, _fit, _plot)
