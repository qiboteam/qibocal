from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.figure_factory as ff
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

    def __getitem__(self, qubits):
        return {
            index: value
            for index, value in self.data.items()
            if qubits == list(index[: len(index) - 2])
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
        computational_basis = [
            format(i, f"0{len(qubit)}b") for i in range(2 ** len(qubit))
        ]
        for state in computational_basis:
            column = np.zeros(2 ** len(qubit))
            qubit_state_data = {
                index: value
                for index, value in qubit_data.items()
                if index[-2] == state
            }
            for index, value in qubit_state_data.items():
                column[(int(index[-1], 2))] = value / data.nshots
            matrix[:, int(state, 2)] = np.flip(column)

        measurement_matrix[tuple(qubit)] = matrix.tolist()
        try:
            readout_mitigation_matrix[tuple(qubit)] = np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError as e:
            log.warning(f"ReadoutMitigationMatrix: the fitting was not succesful. {e}")
            readout_mitigation_matrix[tuple(qubit)] = np.zeros(
                (2 ** len(qubit), 2 ** len(qubit))
            ).tolist()

    return ReadoutMitigationMatrixResults(
        readout_mitigation_matrix=readout_mitigation_matrix,
        measurement_matrix=measurement_matrix,
    )


def _plot(
    data: ReadoutMitigationMatrixData, fit: ReadoutMitigationMatrixResults, qubit
):
    """Plotting function for readout mitigation matrix."""
    fitting_report = ""
    fig = go.Figure()

    if fit is not None:
        computational_basis = [
            format(i, f"0{len(qubit)}b") for i in range(2 ** len(qubit))
        ]

        x = computational_basis
        y = computational_basis[::-1]

        [X, Y] = np.meshgrid(x, y)

        Z = fit.measurement_matrix[tuple(qubit)]

        fig = ff.create_annotated_heatmap(Z, showscale=True)
        fig.update_layout(
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="State prepared",
            yaxis_title="State measured",
            width=700,
            height=700,
        )

    return [fig], None


readout_mitigation_matrix = Routine(_acquisition, _fit, _plot)
"""Readout mitigation matrix protocol."""
