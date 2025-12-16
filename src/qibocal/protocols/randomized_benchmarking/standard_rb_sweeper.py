from collections import defaultdict

import numpy as np
from qibo import Circuit
from qibo.gates import U3, Unitary
from qibo.transpiler.unitary_decompositions import u3_decomposition
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Pulse,
    PulseSequence,
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
    setup,
)

__all__ = ["standard_rb_sweeper"]


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
        num_vz_angles = num_gates * 3
        sweeper_angles = np.zeros((params.niter, num_vz_angles))

        # Setup pulse sequence first
        sequence = PulseSequence()
        pulses_to_sweep: list[Pulse] = []
        ro_pulses: dict[QubitId, Pulse] = {}

        # As each 1Q Clifford is a U3 gate, we can fix the ZXZXZ decomposition per Clifford
        # and set 3 VZ angles in advance to be swept + 2 RX90 pulses as per the decomposition
        for _ in range(num_gates):
            vz_lam = VirtualZ(phase=0)
            vz_theta = VirtualZ(phase=0)
            vz_phi = VirtualZ(phase=0)
            pulses_to_sweep += [vz_lam, vz_theta, vz_phi]

            for qubit in targets:
                qubit_drive, rx90 = mapper[qubit]
                sequence += (
                    [(qubit_drive, vz_lam)]
                    + rx90
                    + [(qubit_drive, vz_theta)]
                    + rx90
                    + [(qubit_drive, vz_phi)]
                )

        for qubit_id in targets:
            qubit = platform.qubits[qubit_id]
            sequence.align([qubit.drive, qubit.acquisition])

            ro_channel, ro_pulse = platform.natives.single_qubit[qubit_id].MZ()[0]
            ro_pulses[qubit_id] = ro_pulse
            sequence += [(ro_channel, ro_pulse)]

        # We iterate across the requested number of iterations for a given depth
        for iter in range(params.niter):
            # Next, we generate a RB sequence for a given depth
            # and extract the corresponding U3 angles
            circuit = Circuit(nqubits=1)
            random_indexes = []
            for k in range(depth):
                layer, index = rb_gen.layer_gen_single_qubit()
                random_indexes.append(index)
                clifford: U3 = layer if layer.name != "id" else U3(0, 0, 0, 0)
                circuit.add(layer)

                pulse_idx = k * 3

                # U3 = RZ(phi + pi) RX(pi/2) RZ(theta + pi) RX(pi/2) RZ(lam)
                theta, phi, lam = clifford.parameters
                sweeper_angles[iter, pulse_idx] = -lam
                sweeper_angles[iter, pulse_idx + 1] = -(theta + np.pi)
                sweeper_angles[iter, pulse_idx + 2] = -(phi + np.pi)

            # Solve inverse
            theta, phi, lam = u3_decomposition(
                Unitary(circuit.unitary(), 0).dagger().parameters[0], backend
            )

            sweeper_angles[iter, -1] = -(phi + np.pi)
            sweeper_angles[iter, -2] = -(theta + np.pi)
            sweeper_angles[iter, -3] = -lam

            for qubit in targets:
                indexes[(qubit, depth)].append(random_indexes)

        sweepers = [
            Sweeper(parameter=Parameter.phase, values=angles, pulses=[pulse])
            for pulse, angles in zip(pulses_to_sweep, sweeper_angles.transpose())
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
