from dataclasses import dataclass
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
from .stream_rb import (
    NATIVE_GATES_PAIRS,
    find_drive_duration,
    find_measurement_duration,
    generate_program,
)


@dataclass
class QuaQiskitRbParameters(StandardRBParameters):
    interleave_cz: bool = False
    debug: Optional[str] = None
    """Dump QUA script and config in a file with this name."""


def _convert_identity(g: Optional[str]) -> str:
    return "i" if g is None else g


def to_indices(sequence: Sequence) -> list[int]:
    return [
        NATIVE_GATES_PAIRS.index((_convert_identity(g0), _convert_identity(g1)))
        for g0, g1 in sequence
    ]


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
    assert len(targets) == 1
    targets = targets[0]
    assert len(targets) == 2

    data = RB2QData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )
    data.circuits[targets] = []

    gate_indices = []
    for depth in params.depths:
        clifford_indices = np.random.randint(0, NCLIFFORDS, size=(params.niter, depth))
        _, circuits = generate_circuits(clifford_indices, params.interleave_cz)
        gate_indices.extend(to_indices(to_sequence(circuit)) for circuit in circuits)
        data.circuits[targets].extend(ids.tolist() for ids in clifford_indices)

    if params.relaxation_time is None:
        relaxation_time = platform.settings.relaxation_time
    else:
        relaxation_time = params.relaxation_time

    max_depth = max(len(circuit) for circuit in gate_indices)
    program = generate_program(
        platform,
        sorted(targets)[::-1],  # FIXME: This will only work for qw5q_platinum
        ncircuits=len(gate_indices),
        nshots=params.nshots,
        relaxation_time=relaxation_time,
        max_depth=max_depth,
    )

    estimated_duration = estimate_duration(
        gate_indices,
        find_drive_duration(platform, targets[0]),
        find_measurement_duration(platform, targets[0]),
        params.nshots,
        relaxation_time,
    )
    log.info("Estimated duration: %.5f sec" % estimated_duration)

    # FIXME: This will only work for qw5q_platinum
    config = generate_config(
        platform, list(platform.qubits.keys()), sorted(targets)[::-1]
    )

    qmm = platform._controller.manager

    if params.debug is not None:
        with open(params.debug, "w") as file:
            file.write(generate_qua_script(program, config))

    qm = qmm.open_qm(config)
    job = qm.execute(
        program, compiler_options=CompilerOptionArguments(flags=["not-strict-timing"])
    )

    # TODO: Progress bar
    for circuit in gate_indices:
        depth = len(circuit)
        job.push_to_input_stream("depth_input_stream", depth)
        job.push_to_input_stream("gates_input_stream", circuit)

    handles = job.result_handles
    handles.wait_for_all_values()

    state0 = handles.get("state0").fetch_all()
    state1 = handles.get("state1").fetch_all()

    state0 = state0.reshape((len(params.depths), params.niter, -1))
    state1 = state1.reshape((len(params.depths), params.niter, -1))
    for i, depth in enumerate(params.depths):
        samples = ((state0[i] + state1[i]) != 0).astype(np.int32)
        data.register_qubit(
            RBType, (targets[0], targets[1], depth), {"samples": samples}
        )

    return data


qua_standard_rb_2q_qiskit = Routine(_acquisition, _fit, _plot)
