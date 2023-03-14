import numpy as np
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("Shot Frequencies", plots.shot_frequencies_bar_chart)
def state_tomography(
    platform,
    qubits,
    sequence,
    nshots=1024,
    relaxation_time=50000,
):
    """State tomography for two qubits.

    The pulse sequence applied consists of two steps.
    First a state preperation sequence given by the user is applied, which
    prepares the target state. Then one additional pulse may be applied to each
    qubit to rotate the measurement basis.
    Following arXiv:0903.2030, tomography is performed by measuring in 15 different
    basis, which are defined by rotating using all pairs of I, RX90, RY90 and RX
    except (RX, RX).

    An example action runcard for using this routine is the following:

        platform: my_platform_name

        qubits: [1, 2]

        format: csv

        actions:

        state_tomography:
            sequence:
                # [[Pulse Type, Target Qubit]]
                # pulses given in the same row are played in parallel
                - [["RX", 1]]
                - [["RY90", 1], ["RY90", 2]]
                - [["FluxPulse", 2, 30, 0.2782]]
                - [["RY90", 1]]
            nshots: 50000
            relaxation_time: 50000

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        sequence (list): List describing the pulse sequence to be used for state preperation.
            See example for more details.
        nshots (int): Number of shots to perform for each measurement.
        relaxation_time (int): Time to wait (in ns) for the qubit to relax between each shot.
    """
    if len(qubits) != 2:
        raise NotImplementedError("Tomography is only implemented for two qubits.")

    qubit1, qubit2 = min(qubits.keys()), max(qubits.keys())

    # reload instrument settings from runcard
    platform.reload_settings()
    basis_rotations = {
        "I": None,
        "RX": lambda qubit, start: platform.create_RX_pulse(qubit, start),
        "RY": lambda qubit, start: platform.create_RX_pulse(
            qubit, start, relative_phase=np.pi / 2
        ),
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
                total_sequence = PulseSequence()
                # state preperation sequence
                for moment in sequence:
                    start = total_sequence.finish
                    for pulse_description in moment:
                        pulse_type, qubit = pulse_description[:2]
                        if pulse_type == "FluxPulse":
                            # FIXME: Flux pulses should be treated similarly to the
                            # other pulses (read from the runcard)
                            # This is different for now, until we understand exactly
                            # what the flux pulse is doing (in terms of 2q gates)
                            flux_duration, flux_amplitude = pulse_description[2:]
                            total_sequence.add(
                                FluxPulse(
                                    start=start,
                                    duration=flux_duration,
                                    amplitude=flux_amplitude,
                                    shape=Rectangular(),
                                    channel=platform.qubits[qubit2].flux.name,
                                    qubit=qubit,
                                )
                            )
                        elif pulse_type in basis_rotations:
                            total_sequence.add(
                                basis_rotations[pulse_type](qubit, start)
                            )

                # basis rotation sequence
                start = total_sequence.finish
                if basis1 is not None:
                    total_sequence.add(basis1(qubit1, start))
                if basis2 is not None:
                    total_sequence.add(basis2(qubit2, start))
                # measurements
                start = total_sequence.finish
                measure1 = platform.create_MZ_pulse(qubit1, start=start)
                measure2 = platform.create_MZ_pulse(qubit2, start=start)
                total_sequence.add(measure1)
                total_sequence.add(measure2)

                results = platform.execute_pulse_sequence(
                    total_sequence, nshots=nshots, relaxation_time=relaxation_time
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
