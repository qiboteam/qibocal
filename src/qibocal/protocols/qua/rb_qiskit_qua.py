from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qiskit import QuantumCircuit
from qm import generate_qua_script, qua

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRBParameters,
    _fit,
    _plot,
)
from qibocal.protocols.randomized_benchmarking.utils import RB2QData, RBType

from .clifford_utils import CliffordUtils, num_from_2q_circuit
from .configuration import generate_config

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


def generate_circuits(
    indices: npt.NDArray,
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
class Elements:
    drive0: str
    drive1: str
    flux: str


@dataclass
class Calibration:
    phase0: float
    phase1: float
    rx_duration: float
    cz_duration: float


@dataclass
class Readout:
    name: str
    threshold: float
    angle: float


@dataclass
class Acquisition:
    readout: Readout
    operation: str = "measure"
    i: Any = field(default_factory=lambda: qua.declare(qua.fixed))
    q: Any = field(default_factory=lambda: qua.declare(qua.fixed))
    state: Any = field(default_factory=lambda: qua.declare(bool))
    ist: Any = field(default_factory=qua.declare_stream)
    qst: Any = field(default_factory=qua.declare_stream)
    state_st: Any = field(default_factory=qua.declare_stream)

    def measure(self):
        qua.measure(
            self.operation,
            self.readout.name,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )

    def save(self):
        cos = np.cos(self.readout.angle)
        sin = np.sin(self.readout.angle)
        qua.assign(self.state, self.i * cos - self.q * sin > self.readout.threshold)
        qua.save(self.state, self.state_st)


def measure(*acquisitions: Acquisition):
    qua.align()
    for acquisition in acquisitions:
        acquisition.measure()
    for acquisition in acquisitions:
        acquisition.save()


def reset(relaxation_time: int, elements: Elements):
    qua.reset_frame(elements.drive0)
    qua.reset_frame(elements.drive1)
    qua.align()
    qua.wait(relaxation_time)


def play_gate(gate: Optional[str], drive: str, duration: float):
    if gate is None:
        qua.wait(duration, drive)
    else:
        qua.play(gate, drive)


def play_sequence(
    sequence: Sequence,
    elements: Elements,
    calibration: Calibration,
):
    drive0 = elements.drive0
    drive1 = elements.drive1
    for pair in sequence:
        if pair == ("cz", "cz"):
            qua.play("cz", elements.flux)
            qua.frame_rotation_2pi(calibration.phase0, drive0)
            qua.frame_rotation_2pi(calibration.phase1, drive1)
            # qua.wait(calibration.cz_duration, drive0, drive1)
            qua.align(drive0, drive1, elements.flux)
        else:
            duration = calibration.rx_duration
            play_gate(pair[0], drive0, duration)
            play_gate(pair[1], drive1, duration)
            qua.wait(duration, elements.flux)


def qua_program(
    sequences: list[Sequence],
    elements: Elements,
    readout0: Readout,
    readout1: Readout,
    calibration: Calibration,
    nshots: int,
    relaxation_time: int,
):
    with qua.program() as experiment:
        n = qua.declare(int)  # shots counter
        acquisition0 = Acquisition(readout0)
        acquisition1 = Acquisition(readout1)
        with qua.for_(n, 0, n < nshots, n + 1):
            for sequence in sequences:
                reset(relaxation_time, elements)
                play_sequence(sequence, elements, calibration)
                measure(acquisition0, acquisition1)

        with qua.stream_processing():
            nseq = len(sequences)
            acquisition0.state_st.boolean_to_int().buffer(nseq).buffer(nshots).save(
                "state0"
            )
            acquisition1.state_st.boolean_to_int().buffer(nseq).buffer(nshots).save(
                "state1"
            )

    return experiment


@dataclass
class QiskitQuaRbParameters(StandardRBParameters):
    batch_size: int = 10
    script_file: Optional[str] = None
    simulation: bool = False


def _acquisition(
    params: QiskitQuaRbParameters, platform: Platform, targets: list[QubitId]
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

    cz_sequence, phases = platform.pairs[(qubit0, qubit1)].native_gates.CZ.sequence()
    if cz_sequence[0].qubit != qubit0:
        assert cz_sequence[0].qubit == qubit1
        qubit0, qubit1 = qubit1, qubit0

    rx0 = platform.qubits[qubit0].native_gates.RX
    rx1 = platform.qubits[qubit1].native_gates.RX
    assert rx0.duration == rx1.duration
    calibration = Calibration(
        phase0=-phases[qubit0] / (2 * np.pi),
        phase1=-phases[qubit1] / (2 * np.pi),
        rx_duration=rx0.duration // 4,
        cz_duration=cz_sequence[0].duration // 4 + 1,
    )

    elements = Elements(
        drive0=f"drive{qubit0}",
        drive1=f"drive{qubit1}",
        flux=f"flux{qubit0}",
    )

    readout = [
        Readout(
            name=f"readout{qubit}",
            threshold=platform.qubits[qubit].threshold,
            angle=platform.qubits[qubit].iq_angle,
        )
        for qubit in [qubit0, qubit1]
    ]

    data = RB2QData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )

    if not params.simulation:
        controller = platform._controller
        manager = controller.manager
        config = generate_config(
            platform, list(platform.qubits.keys()), targets=[qubit0, qubit1]
        )
        machine = manager.open_qm(config)

    for depth in params.depths:
        indices = np.random.randint(0, NCLIFFORDS, size=(params.niter, depth))
        _, circuits = generate_circuits(indices)

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
            batch_size = params.batch_size
            sequences = [to_sequence(circuit) for circuit in circuits]
            assert len(sequences) % batch_size == 0
            nbatches = len(sequences) // batch_size

            state0 = []
            state1 = []
            for i in range(nbatches):
                program = qua_program(
                    sequences[i * batch_size : (i + 1) * batch_size],
                    elements,
                    readout[0],
                    readout[1],
                    calibration,
                    params.nshots,
                    params.relaxation_time // 4,
                )

                if params.script_file is not None:
                    script = generate_qua_script(program, config)
                    with open(params.script_file, "w") as file:
                        file.write(script)

                job = machine.execute(program)
                handles = job.result_handles
                handles.wait_for_all_values()

                state0.append(handles.get("state0").fetch_all().T)
                state1.append(handles.get("state1").fetch_all().T)

            state0 = np.concatenate(state0, axis=0)
            state1 = np.concatenate(state1, axis=0)

        # samples: (niter, nshots)
        samples = ((state0 + state1) != 0).astype(np.int32)
        data.register_qubit(
            RBType, (targets[0][0], targets[0][1], depth), {"samples": samples}
        )

    return data


rb_qiskit_qua = Routine(_acquisition, _fit, _plot)
