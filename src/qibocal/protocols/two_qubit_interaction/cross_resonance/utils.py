from enum import Enum

from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular

from ....auto.operation import QubitId


class SetControl(str, Enum):
    """Helper to create sequence with control set to X or I."""

    Id = "Id"
    X = "X"


def cr_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    setup: SetControl,
    amplitude: float,
    duration: int,
    interpolated_sweeper: bool = False,
) -> tuple[PulseSequence, Pulse, list[Delay]]:
    """CR sequence"""

    sequence = PulseSequence()
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]
    cr_channel = platform.qubit_pairs[control, target].drive
    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=amplitude,
        envelope=Rectangular(),
    )
    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    ro_channel, ro_pulse = natives_target.MZ()[0]
    ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
    if setup == "X":
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((ro_channel, Delay(duration=control_drive_pulse.duration)))
        sequence.append(
            (ro_channel_control, Delay(duration=control_drive_pulse.duration))
        )
        sequence.append((cr_channel, Delay(duration=control_drive_pulse.duration)))

    sequence.append((cr_channel, cr_drive_pulse))

    delay1 = Delay(duration=cr_drive_pulse.duration)
    delay2 = Delay(duration=cr_drive_pulse.duration)
    if interpolated_sweeper:
        sequence.align([cr_channel, ro_channel, ro_pulse_control])
    else:
        sequence.append((ro_channel, delay1))
        sequence.append((ro_channel_control, delay2))

    sequence.append((ro_channel, ro_pulse))
    sequence.append((ro_channel_control, ro_pulse_control))

    return sequence, cr_drive_pulse, [delay1, delay2]
