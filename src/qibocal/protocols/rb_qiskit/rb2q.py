from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, Delay, Platform, PulseSequence
from qiskit import QuantumCircuit

from qibocal.auto.operation import QubitId, Routine
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRBParameters,
    _fit,
    _plot,
)
from qibocal.protocols.randomized_benchmarking.utils import RB2QData, RBType

from .clifford_utils import CliffordUtils, num_from_2q_circuit

INVERSE = np.load(Path(__file__).parent / "data" / "clifford_inverse_2q.npz")["table"]
NCLIFFORDS = len(INVERSE)

BASIS_GATES = ("rx", "ry", "cz")

_PI = np.round(np.pi, decimals=10)
_PI2 = np.round(np.pi / 2, decimals=10)
GATE_MAP = {
    ("rx", _PI): "x180",
    ("ry", _PI): "y180",
    ("rx", _PI2): "x90",
    ("ry", _PI2): "y90",
    ("rx", -_PI2): "-x90",
    ("ry", -_PI2): "-y90",
}

Sequence = list[tuple[Optional[str], Optional[str]]]


def to_sequence(circuit) -> Sequence:
    """Converts Qiskit circuit to list of gate pairs."""
    sequence = [[], []]

    def _synchronize():
        seq1, seq2 = sequence
        n = max(len(seq1), len(seq2))
        seq1 += (n - len(seq1)) * [None]
        seq2 += (n - len(seq2)) * [None]

    for gate in circuit:
        name = gate.operation.name
        if name == "cz":
            _synchronize()
            sequence[0].append("cz")
            sequence[1].append("cz")
        else:
            param = np.round(gate.operation.params[0], decimals=10)
            qubit = gate.qubits[0]._index
            sequence[qubit].append(GATE_MAP[(name, param)])

    _synchronize()
    return [(g1, g2) for g1, g2 in zip(*sequence)]


def _sync_channel(seq: PulseSequence, channel: str, start: float) -> PulseSequence:
    delay = start - seq.channel_duration(channel)
    if delay > 0:
        seq.append((channel, Delay(duration=delay)))
    return seq


def to_qibolab_sequence(
    sequence: Sequence, platform: Platform, qubit0: QubitId, qubit1: QubitId
) -> PulseSequence:
    drive0 = platform.qubits[qubit0].drive
    drive1 = platform.qubits[qubit1].drive
    flux = platform.qubits[qubit0].flux
    natives = platform.natives.single_qubit

    seq = PulseSequence()
    for pair in sequence:
        start = seq.duration
        if pair[0] == "cz":
            _sync_channel(seq, flux, start)
            seq += platform.natives.two_qubit[(qubit0, qubit1)].CZ
        else:
            for i, gate in enumerate(pair):
                q = [qubit0, qubit1][i]
                drive = [drive0, drive1][i]
                _sync_channel(seq, drive, start)
                if gate == "x180":
                    seq += natives[q].R(theta=np.pi, phi=0)
                elif gate == "y180":
                    seq += natives[q].R(theta=np.pi, phi=np.pi / 2)
                elif gate == "x90":
                    seq += natives[q].R(theta=np.pi / 2, phi=0)
                elif gate == "y90":
                    seq += natives[q].R(theta=np.pi / 2, phi=np.pi / 2)
                elif gate == "-x90":
                    seq += natives[q].R(theta=np.pi / 2, phi=np.pi)
                elif gate == "-y90":
                    seq += natives[q].R(theta=np.pi / 2, phi=-np.pi / 2)

    seq |= natives[qubit0].MZ + natives[qubit1].MZ
    return seq


def generate_circuits(
    indices: npt.NDArray,
    interleave_cz: bool = False,
) -> tuple[list[QuantumCircuit], list[QuantumCircuit]]:
    """Generate circuits from Clifford indices.

    Indices must be an array of shape ``(nreps, depth)``.
    """
    clifford = CliffordUtils.clifford_2_qubit_circuit
    circuits = []
    native_circuits = []
    for ids in indices:
        circuit = QuantumCircuit(2)
        native_circuit = QuantumCircuit(2)
        for i in ids:
            circuit = circuit.compose(clifford(i))
            native_circuit = native_circuit.compose(
                clifford(i, basis_gates=BASIS_GATES)
            )
            if interleave_cz:
                circuit.cz(0, 1)
                native_circuit.cz(0, 1)

        # add inverse to circuits
        inverse_index = INVERSE[num_from_2q_circuit(circuit)]
        circuit = circuit.compose(clifford(inverse_index))
        native_circuit = native_circuit.compose(
            clifford(inverse_index, basis_gates=BASIS_GATES)
        )

        circuits.append(circuit)
        native_circuits.append(native_circuit)

    return circuits, native_circuits


@dataclass
class QiskitRbParameters(StandardRBParameters):
    # batch_size: int = 10
    simulation: bool = False
    interleave_cz: bool = False


def _acquisition(
    params: QiskitRbParameters, platform: Platform, targets: list[QubitId]
) -> RB2QData:
    if params.seed is not None:
        np.random.seed(params.seed)

    assert len(targets) == 1
    qubit0, qubit1 = targets[0]
    try:
        qubit0 = int(qubit0)
        qubit1 = int(qubit1)
    except ValueError:
        pass

    cz_sequence = platform.natives.two_qubit[(qubit0, qubit1)].CZ
    flux_channel = cz_sequence[0][0]
    # Make `qubit0` always the one that the flux pulse is applied
    if flux_channel != platform.qubits[qubit0].flux:
        assert flux_channel == platform.qubits[qubit1].flux
        qubit0, qubit1 = qubit1, qubit0

    data = RB2QData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )
    data.circuits[targets[0]] = []

    for depth in params.depths:
        indices = np.random.randint(0, NCLIFFORDS, size=(params.niter, depth))
        _, circuits = generate_circuits(indices, params.interleave_cz)

        if params.simulation:
            from qiskit_aer import AerSimulator

            backend = AerSimulator()

            state0 = np.empty((params.niter, params.nshots), dtype=np.int32)
            state1 = np.empty((params.niter, params.nshots), dtype=np.int32)
            for i, circuit in enumerate(circuits):
                mcircuit = QuantumCircuit(2, 2)
                mcircuit = mcircuit.compose(circuit)
                mcircuit.measure(range(2), range(2))

                job = backend.run(mcircuit, shots=params.nshots)
                result = job.result()
                counts = result.get_counts()

                shots = []
                for key, value in counts.items():
                    shots += value * [[int(x) for x in key]]
                shots = np.array(shots)

                state0[i] = shots[:, 0]
                state1[i] = shots[:, 1]

        else:
            # batch_size = params.batch_size
            sequences = [
                to_qibolab_sequence(to_sequence(circuit), platform, qubit0, qubit1)
                for circuit in circuits
            ]
            # assert len(sequences) % batch_size == 0
            # nbatches = len(sequences) // batch_size
            # for i in range(nbatches):
            # pulse_sequences = to_qibolab_sequence(
            #    sequences[i * batch_size : (i + 1) * batch_size],
            # )

            acquisition0 = platform.qubits[qubit0].acquisition
            acquisition1 = platform.qubits[qubit1].acquisition
            state0 = []
            state1 = []
            for sequence in sequences:
                results = platform.execute(
                    [sequence],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                )
                ro_pulse0 = list(sequence.channel(acquisition0))[-1]
                ro_pulse1 = list(sequence.channel(acquisition1))[-1]
                state0.append(results[ro_pulse0.id])
                state1.append(results[ro_pulse1.id])

            state0 = np.stack(state0)
            state1 = np.stack(state1)

        # samples: (niter, nshots)
        samples = ((state0 + state1) != 0).astype(np.int32)
        data.register_qubit(
            RBType, (targets[0][0], targets[0][1], depth), {"samples": samples}
        )
        data.circuits[targets[0]].extend(ids.tolist() for ids in indices)

    return data


rb_qiskit = Routine(_acquisition, _fit, _plot)
