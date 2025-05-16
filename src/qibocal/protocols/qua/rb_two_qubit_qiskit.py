from dataclasses import dataclass
from typing import Optional

import numpy as np
from qibolab import Platform

from qibocal.auto.operation import QubitId, Routine
from qibocal.protocols.randomized_benchmarking.standard_rb_2q import (
    StandardRBParameters,
    _fit,
    _plot,
)
from qibocal.protocols.randomized_benchmarking.utils import RB2QData, RBType
from qibocal.protocols.rb_qiskit.rb2q import (
    NCLIFFORDS,
    generate_circuits,
    to_sequence,
)

from .stream_circuits import execute


@dataclass
class QuaQiskitRbParameters(StandardRBParameters):
    interleave_cz: bool = False
    batch_size: Optional[int] = None
    debug: Optional[str] = None
    """Dump QUA script and config in a file with this name."""


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

    sequences = []
    for depth in params.depths:
        clifford_indices = np.random.randint(0, NCLIFFORDS, size=(params.niter, depth))
        _, circuits = generate_circuits(clifford_indices, params.interleave_cz)
        sequences.extend(to_sequence(circuit) for circuit in circuits)
        data.circuits[targets].extend(ids.tolist() for ids in clifford_indices)

    state0, state1 = execute(
        sequences,
        platform,
        targets,
        params.nshots,
        params.relaxation_time,
        params.batch_size,
        params.debug,
    )

    state0 = state0.reshape((len(params.depths), params.niter, -1))
    state1 = state1.reshape((len(params.depths), params.niter, -1))
    for i, depth in enumerate(params.depths):
        samples = ((state0[i] + state1[i]) != 0).astype(np.int32)
        data.register_qubit(
            RBType, (targets[0], targets[1], depth), {"samples": samples}
        )
    return data


qua_standard_rb_2q_qiskit = Routine(_acquisition, _fit, _plot)
