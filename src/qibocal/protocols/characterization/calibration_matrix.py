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

from .chsh.utils import calculate_frequencies


@dataclass
class CalibrationMatrixParameters(Parameters):
    """Calibration matrix inputs."""

    pulses: Optional[bool] = True
    """Get calibration matrix using pulses. If False gates will be used."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    bitflip_probabilities: Optional[list[int]] = None
    """Readout error model."""


@dataclass
class CalibrationMatrixResults(Results):
    calibration_matrix: dict[tuple[QubitId, ...], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )

    def save(self, path):
        np.savez(
            path / "calibration_matrix.npy",
            **{str(i): self.calibration_matrix[i] for i in self.calibration_matrix},
        )


@dataclass
class CalibrationMatrixData(Data):
    """CalibrationMatrix acquisition outputs."""

    nqubits: int
    nshots: int
    qubits_ids: list[QubitId]
    data: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, state, freqs):
        for result_state, freq in freqs.items():
            if state not in self.data:
                self.data[state] = {}
            self.data[state][result_state] = freq


def _acquisition(
    params: CalibrationMatrixParameters,
    platform: Platform,
    qubits: Qubits,
) -> CalibrationMatrixData:
    # conversion from list to list in list to make only one plot
    nqubits = len(qubits)
    data = CalibrationMatrixData(
        nqubits=nqubits, nshots=params.nshots, qubits_ids=tuple(qubits)
    )
    for i in range(2**nqubits):
        state = format(i, f"0{nqubits}b")
        if params.pulses:
            sequence = PulseSequence()
            for q, bit in enumerate(state):
                if bit == "1":
                    sequence.add(platform.create_RX_pulse(q, start=0, relative_phase=0))
            measurement_start = sequence.finish
            for q in range(len(state)):
                MZ_pulse = platform.create_MZ_pulse(q, start=measurement_start)
                sequence.add(MZ_pulse)
            results = platform.execute_pulse_sequence(
                sequence, ExecutionParameters(nshots=params.nshots)
            )
            data.add(state, calculate_frequencies(results))
        else:
            c = Circuit(platform.nqubits)
            for q, bit in enumerate(state):
                if bit == "1":
                    c.add(gates.X(q))
                if params.bitflip_probabilities is not None:
                    c.add(
                        gates.M(
                            q,
                            p0=params.bitflip_probabilities[0],
                            p1=params.bitflip_probabilities[1],
                        )
                    )
                else:
                    c.add(gates.M(q))

            results = c(nshots=params.nshots)

            data.add(state, dict(results.frequencies()))

    return data


def _fit(data: CalibrationMatrixData) -> CalibrationMatrixResults:
    calibration_matrix = {}
    matrix = np.zeros((2**data.nqubits, 2**data.nqubits))

    for state, state_data in data.data.items():
        column = np.zeros(2**data.nqubits)
        for basis, basis_freq in state_data.items():
            column[(int(basis, 2))] = basis_freq / data.nshots
        matrix[:, int(state, 2)] = column

    calibration_matrix[data.qubits_ids] = matrix
    return CalibrationMatrixResults(calibration_matrix=calibration_matrix)


def _plot(data: CalibrationMatrixData, fit: CalibrationMatrixResults):
    """Plotting function for Flipping."""
    figures = []
    fitting_report = "No fitting data"
    fig = go.Figure()
    computational_basis = [
        format(i, f"0{data.nqubits}b") for i in range(2**data.nqubits)
    ]
    fig.add_trace(
        go.Heatmap(
            x=computational_basis,
            y=computational_basis[::-1],
            z=fit.calibration_matrix[data.qubits_ids],
            coloraxis="coloraxis",
        ),
    )
    fig.update_layout(
        title="Calibration Matrix",
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)
    fig = go.Figure()
    for state, state_data in data.data.items():
        fig.add_trace(
            go.Bar(
                x=list(state_data.keys()),
                y=list(state_data.values()),
                opacity=1,
                name=state,
                showlegend=True,
            ),
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="State Prepared",
        yaxis_title="Shots",
    )
    figures.append(fig)
    return figures, fitting_report


calibration_matrix = Routine(_acquisition, _fit, _plot, repeat_plot_per_qubit=False)
"""Flipping Routine  object."""
