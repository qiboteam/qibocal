from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.express as px
from qibo import gates
from qibo.backends import GlobalBackend
from qibo.models import Circuit
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.auto.transpile import dummy_transpiler, execute_transpiled_circuit
from qibocal.config import log


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


@dataclass
class ReadoutMitigationMatrixData(Data):
    """ReadoutMitigationMatrix acquisition outputs."""

    qubit_list: list[QubitId]
    """List of qubit ids"""
    nshots: int
    """Number of shots"""
    data: dict = field(default_factory=dict)
    """Raw data acquited."""


def _acquisition(
    params: ReadoutMitigationMatrixParameters,
    platform: Platform,
    targets: list[list[QubitId]],
) -> ReadoutMitigationMatrixData:
    data = ReadoutMitigationMatrixData(
        nshots=params.nshots, qubit_list=[list(qq) for qq in targets]
    )
    backend = GlobalBackend()
    backend.platform = platform
    transpiler = dummy_transpiler(backend)
    qubit_map = [i for i in range(platform.nqubits)]
    for qubits in targets:
        nqubits = len(qubits)
        for i in range(2**nqubits):
            state = format(i, f"0{nqubits}b")
            c = Circuit(
                platform.nqubits,
            )
            for q, bit in enumerate(state):
                if bit == "1":
                    c.add(gates.X(qubits[q]))
            c.add(gates.M(*[qubits[i] for i in range(len(state))]))
            _, results = execute_transpiled_circuit(
                c, qubit_map, backend, nshots=params.nshots, transpiler=transpiler
            )
            frequencies = np.zeros(2 ** len(qubits))
            for state, freq in results.frequencies().items():
                frequencies[int(state, 2)] = freq
            for freq in frequencies:  # TODO: Remove this loop?
                data.register_qubit(
                    ReadoutMitigationMatrixType,
                    (qubits),
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
        mitigation_matrix = np.vstack(mitigation_matrix)
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
        computational_basis = [
            format(i, f"0{len(target)}b") for i in range(2 ** len(target))
        ]
        z = np.array(fit.readout_mitigation_matrix[tuple(target)]) * data.nshots
        fig = px.imshow(
            z,
            x=computational_basis,
            y=computational_basis,
            text_auto=True,
            labels={
                "x": "Prepeared States",
                "y": "Measured States",
                "color": "Probabilities",
            },
            width=700,
            height=700,
        )
        figs.append(fig)
    return figs, fitting_report


readout_mitigation_matrix = Routine(_acquisition, _fit, _plot)
"""Readout mitigation matrix protocol."""
