from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import plotly.express as px
from qibo import gates
from qibo.backends import get_backend
from qibo.models import Circuit
from qibolab import PulseSequence
from scipy.sparse import lil_matrix

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .utils import calculate_frequencies, computational_basis


@dataclass
class ReadoutMitigationMatrixParameters(Parameters):
    """ReadoutMitigationMatrix matrix inputs."""

    pulses: Optional[bool] = True
    """Get readout mitigation matrix using pulses. If False gates will be used."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time [ns]."""


@dataclass
class ReadoutMitigationMatrixResults(Results):
    readout_mitigation_matrix: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = (
        field(default_factory=dict)
    )
    """Readout mitigation matrices (inverse of measurement matrix)."""
    measurement_matrix: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Matrix containing measurement matrices for each state."""


ReadoutMitigationMatrixId = tuple[Tuple[QubitId, ...], str, str]
"""Data identifier for single list of qubits.

Tuple[QubitId, ...] is the qubits which have been passed on as parameters.
The two strings represents the expected state and the measured state.
"""


@dataclass
class ReadoutMitigationMatrixData(Data):
    """ReadoutMitigationMatrix acquisition outputs."""

    qubit_list: list[QubitId]
    """List of qubit ids"""
    nshots: int
    """Number of shots"""
    data: dict[ReadoutMitigationMatrixId, float] = field(default_factory=dict)
    """Raw data acquited."""

    def add(self, qubits: list[QubitId], state: str, freqs: dict[str, int]):
        """Adding frequency to data."""

        for result_state, freq in freqs.items():
            self.data[
                qubits
                + (
                    state,
                    result_state,
                )
            ] = freq

        for basis in [format(i, f"0{len(qubits)}b") for i in range(2 ** len(qubits))]:
            if (
                qubits
                + (
                    state,
                    basis,
                )
                not in self.data
            ):
                self.data[
                    qubits
                    + (
                        state,
                        basis,
                    )
                ] = 0

    def matrix(self, qubits: list[QubitId]):
        """Retrieve data for single qubits list."""

        matrix = np.zeros((2 ** len(qubits), 2 ** len(qubits)))
        for state in computational_basis(len(qubits)):
            column = np.zeros(2 ** len(qubits))
            qubit_state_data = {
                index: value
                for index, value in self.data.items()
                if index[-2] == state and qubits == list(index[: len(index) - 2])
            }
            for index, value in qubit_state_data.items():
                column[(int(index[-1], 2))] = value / self.nshots
            matrix[:, int(state, 2)] = np.flip(column)
        return matrix


def _acquisition(
    params: ReadoutMitigationMatrixParameters,
    platform: CalibrationPlatform,
    targets: list[list[QubitId]],
) -> ReadoutMitigationMatrixData:
    data = ReadoutMitigationMatrixData(
        nshots=params.nshots, qubit_list=[list(qq) for qq in targets]
    )
    backend = get_backend()
    backend.platform = platform
    transpiler = dummy_transpiler(backend)

    for qubits in targets:
        nqubits = len(qubits)
        for state in computational_basis(nqubits):
            if params.pulses:
                sequence = PulseSequence()
                ro_pulses = {}
                for q, bit in enumerate(state):
                    natives = platform.natives.single_qubit[qubits[q]]
                    if bit == "1":
                        sequence |= natives.RX()
                    sequence |= natives.MZ()
                    ro_pulses[qubits[q]] = list(
                        sequence.channel(platform.qubits[qubits[q]].acquisition)
                    )[-1]
                results = platform.execute([sequence], nshots=params.nshots)
                data.add(
                    tuple(qubits), state, calculate_frequencies(results, ro_pulses)
                )
            else:
                c = Circuit(len(qubits))
                for q, bit in enumerate(state):
                    if bit == "1":
                        c.add(gates.X(q))
                c.add(gates.M(*[i for i in range(len(state))]))
                _, results = execute_transpiled_circuit(
                    c, qubits, backend, nshots=params.nshots, transpiler=transpiler
                )
                data.add(tuple(qubits), state, dict(results.frequencies()))
    return data


def _fit(data: ReadoutMitigationMatrixData) -> ReadoutMitigationMatrixResults:
    """Post processing for readout mitigation matrix protocol."""
    readout_mitigation_matrix = {}
    measurement_matrix = {}
    for qubit in data.qubit_list:
        matrix = data.matrix(qubit)
        measurement_matrix[tuple(qubit)] = matrix.tolist()
        try:
            readout_mitigation_matrix[tuple(qubit)] = np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError as e:
            log.warning(f"ReadoutMitigationMatrix: the fitting was not succesful. {e}")

    return ReadoutMitigationMatrixResults(
        readout_mitigation_matrix=readout_mitigation_matrix,
        measurement_matrix=measurement_matrix,
    )


def _plot(
    data: ReadoutMitigationMatrixData,
    fit: ReadoutMitigationMatrixResults,
    target: list[QubitId],
):
    """Plotting function for readout mitigation matrix."""
    fitting_report = ""
    figs = []
    if fit is not None:
        basis = computational_basis(len(target))
        z = fit.measurement_matrix[tuple(target)]

        fig = px.imshow(
            z,
            x=basis,
            y=basis[::-1],
            text_auto=True,
            labels={
                "x": "Prepared States",
                "y": "Measured States",
                "color": "Probabilities",
            },
            width=700,
            height=700,
        )
        figs.append(fig)
    return figs, fitting_report


def _update(
    results: ReadoutMitigationMatrixData,
    platform: CalibrationPlatform,
    target: list[QubitId],
):
    # create empty matrix if it doesn't exist
    if platform.calibration.readout_mitigation_matrix is None:
        platform.calibration.readout_mitigation_matrix = lil_matrix(
            (2**platform.calibration.nqubits, 2**platform.calibration.nqubits)
        )

    # compute indices
    mask = sum(1 << platform.calibration.qubit_index(i) for i in target)
    indices = [i for i in range(2**platform.calibration.nqubits) if (i & mask) == i]

    # update matrix
    platform.calibration.readout_mitigation_matrix[np.ix_(indices, indices)] = (
        results.readout_mitigation_matrix[tuple(target)]
    )


readout_mitigation_matrix = Routine(_acquisition, _fit, _plot, _update)
"""Readout mitigation matrix protocol."""
