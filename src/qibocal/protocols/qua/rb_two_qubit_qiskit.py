from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
from qibolab import Platform
from qm import CompilerOptionArguments, generate_qua_script

from qibocal.auto.operation import QubitId, Routine
from qibocal.config import log
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRBParameters,
    _fit,
    _plot,
)
from qibocal.protocols.randomized_benchmarking.utils import RB2QData, RBType
from qibocal.protocols.rb_qiskit.rb2q import (
    NCLIFFORDS,
    Sequence,
    generate_circuits,
    to_sequence,
)

from .configuration import generate_config
from .stream_rb import find_drive_duration, find_measurement_duration, generate_program


@dataclass
class QuaQiskitRbParameters(StandardRBParameters):
    debug: Optional[str] = None
    """Dump QUA script and config in a file with this name."""


NATIVE_GATES = ["i", "x180", "y180", "x90", "y90", "-x90", "-y90"]
NATIVE_GATES_PAIRS = list(product(NATIVE_GATES, NATIVE_GATES))
NATIVE_GATES_PAIRS.append(("cz", "cz"))


def to_indices(sequence: Sequence) -> list[int]:
    return [NATIVE_GATES_PAIRS.index(pair) for pair in sequence]


def estimate_duration(
    circuits, rx_duration, mz_duration, nshots, relaxation_time
) -> int:
    duration = sum(len(circuit) * rx_duration for circuit in circuits)
    duration += len(circuits) * (mz_duration + relaxation_time)
    return duration * nshots / 1e9


def _acquisition(
    params: QuaQiskitRbParameters, platform: Platform, targets: list[QubitId]
) -> RB2QData:
    """Data acquisition for two qubit Standard Randomized Benchmarking."""

    data = RB2QData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )
    data.circuits[targets[0]] = []

    gate_indices = []
    for depth in params.depths:
        clifford_indices = np.random.randint(0, NCLIFFORDS, size=(params.niter, depth))
        _, circuits = generate_circuits(clifford_indices)
        gate_indices.extend(to_indices(to_sequence(circuit)) for circuit in circuits)
        data.circuits[targets[0]].extend(ids.tolist() for ids in clifford_indices)

    if params.relaxation_time is None:
        relaxation_time = platform.settings.relaxation_time
    else:
        relaxation_time = params.relaxation_time

    max_depth = max(len(circuit) for circuit in circuits)
    program = generate_program(
        platform,
        targets,
        ncircuits=len(circuits),
        nshots=params.nshots,
        relaxation_time=relaxation_time,
        max_depth=max_depth,
    )

    estimated_duration = estimate_duration(
        circuits,
        find_drive_duration(platform, targets[0]),
        find_measurement_duration(platform, targets[0]),
        params.nshots,
        relaxation_time,
    )
    log.info("Estimated duration: %.5f sec" % estimated_duration)

    config = generate_config(platform, list(platform.qubits.keys()), targets)

    qmm = platform._controller.manager

    if params.debug is not None:
        with open(params.debug, "w") as file:
            file.write(generate_qua_script(program, config))

    qm = qmm.open_qm(config)
    job = qm.execute(
        program, compiler_options=CompilerOptionArguments(flags=["not-strict-timing"])
    )

    # TODO: Progress bar
    for circuit in circuits:
        depth = len(circuit)
        job.push_to_input_stream("depth_input_stream", depth)
        job.push_to_input_stream("gates_input_stream", circuit)

    handles = job.result_handles
    handles.wait_for_all_values()

    state0 = handles.get("state0").fetch_all()
    state1 = handles.get("state1").fetch_all()

    for i, depth in enumerate(params.depths):
        samples = ((state0[i] + state1[i]) != 0).astype(np.int32)
        data.register_qubit(
            RBType, (targets[0][0], targets[0][1], depth), {"samples": samples}
        )

    return data


qua_standard_rb_2q_qiskit = Routine(_acquisition, _fit, _plot)
