import numpy as np
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular

from qibocal.data import DataUnits


def state_tomography(
    platform,
    qubits,
    flux_pulse_duration,  # TODO: Should be read from the runcard
    flux_pulse_amplitude,
    nshots=1024,
    relaxation_time=50000,
):
    if len(qubits) != 2:
        raise NotImplementedError("Tomography is only implemented for two qubits.")

    qubit1, qubit2 = min(qubits.keys()), max(qubits.keys())

    # reload instrument settings from runcard
    platform.reload_settings()
    basis_rotations = {
        "I": None,
        "RX": lambda qubit, start: platform.create_RX_pulse(qubit, start),
        "RX90": lambda qubit, start: platform.create_RX90_pulse(qubit, start),
        "RY90": lambda qubit, start: platform.create_RX90_pulse(
            qubit, start, relative_phase=np.pi / 2
        ),
    }

    data = DataUnits(
        name="data",
        options=[
            "rotation1",
            "rotation2",
            "shots",
            "qubit",
        ],
    )
    for label1, basis1 in basis_rotations.items():
        for label2, basis2 in basis_rotations.items():
            if not (label1 == "RX" and label2 == "RX"):
                sequence = PulseSequence()
                # entangler sequence
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit1, start=0, relative_phase=np.pi / 2
                    )
                )
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit2, start=0, relative_phase=np.pi / 2
                    )
                )
                sequence.add(
                    FluxPulse(
                        start=sequence.finish,
                        duration=flux_pulse_duration,
                        amplitude=flux_pulse_amplitude,
                        shape=Rectangular(),
                        channel=platform.qubits[qubit2].flux.name,
                        qubit=qubit2,
                    )
                )
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit1, start=sequence.finish, relative_phase=np.pi / 2
                    )
                )
                # basis rotation sequence
                start = sequence.finish
                if basis1 is not None:
                    sequence.add(basis1(qubit1, start))
                if basis2 is not None:
                    sequence.add(basis2(qubit2, start))
                # measurements
                start = sequence.finish
                measure1 = platform.create_MZ_pulse(qubit1, start=start)
                measure2 = platform.create_MZ_pulse(qubit2, start=start)
                sequence.add(measure1)
                sequence.add(measure2)

                results = platform.execute_pulse_sequence(
                    sequence, nshots=nshots, relaxation_time=relaxation_time
                )

                # store the results
                result1 = results[measure1.serial]
                result2 = results[measure2.serial]
                r1 = {
                    "rotation1": nshots * [label1],
                    "rotation2": nshots * [label2],
                    "shots": result1.shots,
                    "qubit": nshots * [qubit1],
                }
                r1.update(result1.to_dict(average=False))
                data.add_data_from_dict(r1)

                r2 = {
                    "rotation1": nshots * [label1],
                    "rotation2": nshots * [label2],
                    "shots": result2.shots,
                    "qubit": nshots * [qubit2],
                }
                r2.update(result2.to_dict(average=False))
                data.add_data_from_dict(r2)

                # save data
                yield data
