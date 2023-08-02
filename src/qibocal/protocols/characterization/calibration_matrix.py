from dataclasses import dataclass, field
from typing import Optional

from qibo import gates
from qibo.models import Circuit
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

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
    """Flipping outputs."""


@dataclass
class CalibrationMatrixData(Data):
    """CalibrationMatrix acquisition outputs."""

    nqubits: int
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
    nqubits = len(qubits)
    data = CalibrationMatrixData(nqubits=nqubits)
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
    return CalibrationMatrixResults()


# TODO: fix plotting
def _plot(data: CalibrationMatrixData, fit: CalibrationMatrixResults, qubit):
    """Plotting function for Flipping."""
    print(data)

    return [], ""


calibration_matrix = Routine(_acquisition, _fit, _plot)
"""Flipping Routine  object."""
