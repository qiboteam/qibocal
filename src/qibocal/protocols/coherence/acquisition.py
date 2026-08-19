import numpy as np
from qibolab import Delay, Platform, PulseSequence

from qibocal.auto.operation import QubitId

CoherenceType = np.dtype(
    [("wait", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for coherence routines."""


def average_single_shots(data_type, single_shots):
    """Convert single shot acquisition results of signal routines to averaged.

    Args:
        data_type: Type of produced data object (eg. ``T1SignalData``, ``T2SignalData`` etc.).
        single_shots (dict): Dictionary containing acquired single shot data.
    """
    data = data_type()
    for qubit, values in single_shots.items():
        data.register_qubit(
            CoherenceType,
            (qubit),
            {name: values[name].mean(axis=0) for name in values.dtype.names},
        )
    return data


def dynamical_decoupling_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
    n: int = 1,
    kind: str = "CPMG",
) -> tuple[PulseSequence, list[Delay]]:
    """Create dynamical decoupling sequence.

    Two sequences are available:
    - CP: RX90 (wait RX wait )^N RX90
    - CPMG: RX90 (wait RY wait )^N RX90
    """

    assert kind in ["CPMG", "CP"], f"Unknown sequence {kind}, please use CP or CPMG"
    sequence = PulseSequence()
    all_delays = []
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel = platform.qubits[qubit].drive
        rx90_sequence = natives.R(theta=np.pi / 2)
        decoupling_sequence = (
            natives.R(phi=np.pi / 2) if kind == "CPMG" else natives.RX()
        )
        ro_channel, ro_pulse = natives.MZ()[0]

        drive_delays = 2 * n * [Delay(duration=wait)]
        ro_delays = 2 * n * [Delay(duration=wait)]

        sequence += rx90_sequence

        for i in range(n):
            sequence.append((qd_channel, drive_delays[2 * i]))
            sequence.append((ro_channel, ro_delays[2 * i]))
            sequence += decoupling_sequence
            sequence.append((qd_channel, drive_delays[2 * i + 1]))
            sequence.append((ro_channel, ro_delays[2 * i + 1]))

        sequence += rx90_sequence

        sequence.append(
            (
                ro_channel,
                Delay(
                    duration=2 * rx90_sequence.duration
                    + n * decoupling_sequence.duration
                ),
            )
        )

        sequence.append((ro_channel, ro_pulse))
        all_delays.extend(drive_delays)
        all_delays.extend(ro_delays)
    return sequence, all_delays
