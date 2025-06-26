from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import plotly.express as px
from qibo import gates
from qibo.backends import construct_backend
from qibo.models import Circuit

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

__all__ = ["readout_mitigation_matrix"]


@dataclass
class ReadoutMitigationMatrixParameters(Parameters):
    """ReadoutMitigationMatrix matrix inputs."""

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


ReadoutMitigationMatrixType = np.dtype(
    [
        ("state", int),
        ("frequency", np.float64),
    ]
)


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


def _acquisition(
    params: ReadoutMitigationMatrixParameters,
    platform: CalibrationPlatform,
    targets: list[list[QubitId]],
) -> ReadoutMitigationMatrixData:
    data = ReadoutMitigationMatrixData(
        nshots=params.nshots, qubit_list=[list(qq) for qq in targets]
    )
    backend = construct_backend("qibolab", platform=platform)
    transpiler = dummy_transpiler(backend)

    for qubits in targets:
        nqubits = len(qubits)
        for i in range(2**nqubits):
            state = format(i, f"0{nqubits}b")
            c = Circuit(
                nqubits,
            )
            for q, bit in enumerate(state):
                if bit == "1":
                    c.add(gates.X(q))
            c.add(gates.M(*range(nqubits)))
            _, results = execute_transpiled_circuit(
                c, qubits, backend, nshots=params.nshots, transpiler=transpiler
            )
            frequencies = np.zeros(2 ** len(qubits))
            for i, freq in results.frequencies().items():
                frequencies[int(i, 2)] = freq
            for freq in frequencies:
                data.register_qubit(
                    ReadoutMitigationMatrixType,
                    (tuple(qubits)),
                    dict(
                        state=np.array([int(state, 2)]),
                        frequency=freq,
                    ),
                )
    return data


def _fit(data: ReadoutMitigationMatrixData) -> ReadoutMitigationMatrixResults:
    """Post processing for readout mitigation matrix protocol."""
    readout_mitigation_matrix = {}
    for qubits in data.qubit_list:
        qubit_data = data.data[tuple(qubits)]
        mitigation_matrix = []
        for state in range(2 ** len(qubits)):
            mitigation_matrix.append(qubit_data[qubit_data.state == state].frequency)
        mitigation_matrix = np.vstack(mitigation_matrix) / data.nshots
        try:
            readout_mitigation_matrix[tuple(qubits)] = np.linalg.inv(
                mitigation_matrix
            ).tolist()
        except np.linalg.LinAlgError as e:
            log.warning(f"ReadoutMitigationMatrix: the fitting was not succesful. {e}")
    res = ReadoutMitigationMatrixResults(
        readout_mitigation_matrix=readout_mitigation_matrix,
    )

    return res


def _plot(
    data: ReadoutMitigationMatrixData,
    fit: ReadoutMitigationMatrixResults,
    target: list[QubitId],
):
    """Plotting function for readout mitigation matrix."""
    fitting_report = ""
    figs = []
    if fit is not None:
        if tuple(target) in fit.readout_mitigation_matrix:
            computational_basis = [
                format(i, f"0{len(target)}b") for i in range(2 ** len(target))
            ]
            # use pinv since it should be already invertibile
            # however when casting to list we could lose precision
            measurement_matrix = np.linalg.pinv(
                fit.readout_mitigation_matrix[tuple(target)]
            )
            z = measurement_matrix
            fig = px.imshow(
                z,
                x=computational_basis,
                y=computational_basis,
                text_auto=True,
                labels={
                    "x": "Measured States",
                    "y": "Prepared States",
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
    platform.calibration.set_readout_mitigation_matrix_element(
        tuple(target), results.readout_mitigation_matrix
    )


readout_mitigation_matrix = Routine(_acquisition, _fit, _plot, _update)
"""Readout mitigation matrix protocol."""
