from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from .utils import calculate_frequencies


@dataclass
class ReadoutMitigationMatrixParameters(Parameters):
    """ReadoutMitigationMatrix matrix inputs."""

    pulses: Optional[bool] = True
    """Get readout mitigation matrix using pulses. If False gates will be used."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ReadoutMitigationMatrixResults(Results):
    readout_mitigation_matrix: dict[
        tuple[QubitId, ...], npt.NDArray[np.float64]
    ] = field(default_factory=dict)
    """Readout mitigation matrices (inverse of measurement matrix)."""
    measurement_matrix: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Matrix containing measurement matrices for each state."""

    def save(self, path):
        """Store readout mitigation matrix to file."""
        np.savez(
            path / "readout_mitigation_matrix",
            **{
                str(i): self.readout_mitigation_matrix[i]
                for i in self.readout_mitigation_matrix
            },
        )


@dataclass
class ReadoutMitigationMatrixData(Data):
    """ReadoutMitigationMatrix acquisition outputs."""

    qubits_list: list
    """List of qubit ids"""
    nshots: int
    """Number of shots"""
    data: dict = field(default_factory=dict)
    """Raw data acquited."""

    def add(self, qubits, state, freqs):
        for result_state, freq in freqs.items():
            label = qubits + (state,)
            if label not in self.data:
                self.data[label] = {}
            self.data[label][result_state] = freq

        for basis in [format(i, f"0{len(qubits)}b") for i in range(2 ** len(qubits))]:
            if basis not in self.data[qubits + (state,)]:
                self.data[qubits + (state,)][basis] = 0

    def __getitem__(self, qubits):
        return {
            index: value
            for index, value in self.data.items()
            if qubits == list(index[: len(index) - 1])
        }


def _acquisition(
    params: ReadoutMitigationMatrixParameters,
    platform: Platform,
    qubits: list[Qubits],
) -> ReadoutMitigationMatrixData:
    data = ReadoutMitigationMatrixData(
        nshots=params.nshots, qubits_list=[list(qq) for qq in qubits]
    )
    for qubit_list in qubits:
        nqubits = len(qubit_list)
        for i in range(2**nqubits):
            state = format(i, f"0{nqubits}b")
            if params.pulses:
                sequence = PulseSequence()
                for q, bit in enumerate(state):
                    if bit == "1":
                        sequence.add(
                            platform.create_RX_pulse(
                                qubit_list[q], start=0, relative_phase=0
                            )
                        )
                measurement_start = sequence.finish
                for q in range(len(state)):
                    MZ_pulse = platform.create_MZ_pulse(
                        qubit_list[q], start=measurement_start
                    )
                    sequence.add(MZ_pulse)
                results = platform.execute_pulse_sequence(
                    sequence, ExecutionParameters(nshots=params.nshots)
                )
                data.add(qubit_list, state, calculate_frequencies(results, qubit_list))
            else:
                c = Circuit(platform.nqubits)
                for q, bit in enumerate(state):
                    if bit == "1":
                        c.add(gates.X(qubit_list[q]))
                    c.add(gates.M(qubit_list[q]))
                results = c(nshots=params.nshots)

                data.add(qubit_list, state, dict(results.frequencies()))

    return data


def _fit(data: ReadoutMitigationMatrixData) -> ReadoutMitigationMatrixResults:
    """Post processing for readout mitigation matrix protocol."""
    readout_mitigation_matrix = {}
    measurement_matrix = {}
    for qubit in data.qubits_list:
        qubit_data = data[qubit]
        matrix = np.zeros((2 ** len(qubit), 2 ** len(qubit)))

        for label, state_data in qubit_data.items():
            state = label[-1]
            column = np.zeros(2 ** len(qubit))
            for basis, basis_freq in state_data.items():
                column[(int(basis, 2))] = basis_freq / data.nshots
            matrix[:, int(state, 2)] = np.flip(column)

        measurement_matrix[tuple(qubit)] = matrix
        try:
            readout_mitigation_matrix[tuple(qubit)] = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as e:
            log.warning(f"ReadoutMitigationMatrix: the fitting was not succesful. {e}")
            readout_mitigation_matrix[tuple(qubit)] = np.zeros(
                (2 ** len(qubit), 2 ** len(qubit))
            )

    return ReadoutMitigationMatrixResults(
        readout_mitigation_matrix=readout_mitigation_matrix,
        measurement_matrix=measurement_matrix,
    )


def _plot(
    data: ReadoutMitigationMatrixData, fit: ReadoutMitigationMatrixResults, qubit
):
    """Plotting function for readout mitigation matrix."""
    fitting_report = "No fitting data"
    fig = go.Figure()
    computational_basis = [format(i, f"0{len(qubit)}b") for i in range(2 ** len(qubit))]

    fig.add_trace(
        go.Heatmap(
            x=computational_basis,
            y=computational_basis[::-1],
            z=fit.measurement_matrix[tuple(qubit)],
            text_auto=True,
        ),
    )
    fig.update_layout(
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="State prepared",
        yaxis_title="State measured",
    )

    return [fig], fitting_report


readout_mitigation_matrix = Routine(_acquisition, _fit, _plot)
"""Readout mitigation matrix protocol."""
