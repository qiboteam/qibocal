from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class StateTomographyData(Data):
    """Tomography data"""


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
            len(targets), wire_names=[str(i) for i in range(len(targets))]
        )
        params.circuit.add(gates.X(i) for i in targets)

    for basis in ["X", "Y", "Z"]:
        basis_circuit = deepcopy(params.circuit)
        basis_circuit.add(gates.M(i, basis=getattr(gates, basis)) for i in targets)

        simulation[basis] = NumpyBackend().execute_circuit(basis_circuit).samples()
        hardware[basis] = basis_circuit(nshots=params.nshots).samples()

    print(simulation)
    return StateTomographyData()


def _fit(data: StateTomographyData) -> StateTomographyResults:
    """Post-processing for State tomography."""

    return StateTomographyResults()


def _plot(data: StateTomographyData, fit: StateTomographyResults, target: QubitId):
    """Plotting for state tomography"""
    return [], ""


state_tomography = Routine(_acquisition, _fit, _plot)
