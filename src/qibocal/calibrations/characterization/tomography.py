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
    if len(qubits) != 1:
        raise NotImplementedError("Tomography is only implemented for two qubits.")

    qubit = list(qubits.keys())[0]
    qubit1 = min(qubit, 2)
    qubit2 = max(qubit, 2)

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

    data_options = lambda q: [
        f"rotation{q}",
        f"i{q}[V]",
        f"q{q}[V]",
        f"shots{q}",
        f"qubit{q}",
    ]
    data = DataUnits(
        name="data",
        options=data_options(1) + data_options(2),
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
                print(result1.i)
                print(result2.q)
                r = {
                    "rotation1": nshots * [label1],
                    "rotation2": nshots * [label2],
                    "i1[V]": result1.i,
                    "q1[V]": result1.q,
                    "shots1": result1.shots,
                    "i2[V]": result2.i,
                    "q2[V]": result2.q,
                    "shots2": result2.shots,
                    "qubit1": nshots * [qubit1],
                    "qubit2": nshots * [qubit2],
                }
                data.add_data_from_dict(r)

                # save data
                yield data
