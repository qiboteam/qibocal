from collections import defaultdict

import numpy as np
from qibo.gates import U3, Unitary
from qibo.transpiler.unitary_decompositions import u3_decomposition
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    PulseSequence,
    Readout,
    Sweeper,
    VirtualZ,
)

from qibocal.auto.operation import Routine
from qibocal.calibration.calibration import QubitId
from qibocal.calibration.platform import CalibrationPlatform
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    _fit,
    _plot,
    _update,
)
from qibocal.protocols.randomized_benchmarking.utils import (
    RB_Generator,
    RBData,
    RBType,
    add_inverse_layer,
    layer_circuit,
    setup,
)

__all__ = ["standard_rb_sweeper"]

NUM_VZ_PER_CLIFFORD = 3


def _acquisition(
    params: StandardRBParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.
    Instead of using circuits, the Cliffords are manually transpiled
    and the resulting sequence can be swept with sweepers.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params: All parameters in one object.
        platform: CalibrationPlatform the experiment is executed on.
        target: list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    data, backend = setup(params, platform, single_qubit=True)
    rb_gen = RB_Generator(params.seed)

    # Helper map to reuse drive channel and RX90 for the pulse sequence
    mapper = {
        qubit: (
            platform.qubits[qubit].drive,
            platform.natives.single_qubit[qubit].R(np.pi / 2),
        )
        for qubit in targets
    }
    indexes = defaultdict(list)

    for depth in params.depths:
        num_gates = depth + 1
        num_vz_angles = num_gates * NUM_VZ_PER_CLIFFORD
        sweeper_angles = np.zeros((params.niter, num_vz_angles))

        # Setup pulse sequence first
        vz_sweep_pulses = [VirtualZ(phase=0) for _ in range(num_vz_angles)]
        ro_pulses: dict[QubitId, Readout] = {}
        qubitseqs = {q: PulseSequence() for q in targets}

        for idx in range(num_gates):
            pulse_idx = idx * NUM_VZ_PER_CLIFFORD
            vz_lam, vz_theta, vz_phi = vz_sweep_pulses[
                pulse_idx : pulse_idx + NUM_VZ_PER_CLIFFORD
            ]
            for qubit in targets:
                qubit_drive, rx90 = mapper[qubit]
                qubitseqs[qubit] += (
                    [(qubit_drive, vz_lam)]
                    + rx90
                    + [(qubit_drive, vz_theta)]
                    + rx90
                    + [(qubit_drive, vz_phi)]
                )

        for qubit in targets:
            mz = platform.natives.single_qubit[qubit].MZ()
            ro_pulses[qubit] = mz[0][1]
            qubitseqs[qubit] |= mz

        sequence = PulseSequence()
        for qseq in qubitseqs.values():
            sequence += qseq

        # We iterate across the requested number of iterations for a given depth
        for iter in range(params.niter):
            # Next, we generate a RB sequence for a given depth
            # and extract the corresponding U3 angles
            circuit, random_indexes = layer_circuit(rb_gen, depth, 0)
            for idx, layer in enumerate(circuit.queue):
                clifford: U3 = layer if layer.name != "id" else U3(0, 0, 0, 0)

                theta, phi, lam = clifford.parameters
                vz_idx = idx * NUM_VZ_PER_CLIFFORD
                sweeper_angles[iter, vz_idx : vz_idx + NUM_VZ_PER_CLIFFORD] = [
                    -lam,
                    -(theta + np.pi),
                    -(phi + np.pi),
                ]

            add_inverse_layer(circuit, rb_gen)
            inverse_layer: Unitary = circuit.queue[-1]
            theta, phi, lam = u3_decomposition(inverse_layer.parameters[0], backend)
            sweeper_angles[iter, -3:] = [-lam, -(theta + np.pi), -(phi + np.pi)]

            for qubit in targets:
                indexes[(qubit, depth)].append(random_indexes)

        sweepers = [
            Sweeper(parameter=Parameter.phase, values=angles, pulses=[pulse])
            for pulse, angles in zip(vz_sweep_pulses, sweeper_angles.transpose())
        ]
        results = platform.execute(
            [sequence],
            [sweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        )
        for qubit in targets:
            data.register_qubit(
                RBType,
                (qubit, depth),
                dict(
                    samples=results[ro_pulses[qubit].id].transpose(),
                ),
            )

    data.circuits = indexes
    data.npulses_per_clifford = 2
    return data


standard_rb_sweeper = Routine(_acquisition, _fit, _plot, _update)
