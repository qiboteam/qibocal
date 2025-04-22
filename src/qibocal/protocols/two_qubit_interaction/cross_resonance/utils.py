from enum import Enum

import numpy as np
from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular, VirtualZ

from ....auto.operation import QubitId
from ....update import replace


class SetControl(str, Enum):
    """Helper to create sequence with control set to X or I."""

    Id = "Id"
    X = "X"


class Basis(str, Enum):
    """Measurement basis."""

    X = "X"
    Y = "Y"
    Z = "Z"


def cr_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    setup: SetControl,
    amplitude: float,
    duration: int,
    interpolated_sweeper: bool = False,
    echo: bool = False,
    basis: Basis = Basis.Z,
) -> tuple[PulseSequence, list[Pulse], list[Delay]]:
    """CR sequence"""

    if echo:
        assert basis == Basis.Z, "Echo sequence only available in Z basis"

    cr_pulses = []
    sequence = PulseSequence()
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]
    cr_channel = platform.qubits[control].drive_extra[target]
    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=amplitude,
        relative_phase=0,
        envelope=Rectangular(),
    )
    cr_pulses.append(cr_drive_pulse)
    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    target_drive_channel, target_drive_pulse = natives_target.RX()[0]
    ro_channel, ro_pulse = natives_target.MZ()[0]
    ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
    if setup == SetControl.X:
        control_delay = Delay(duration=control_drive_pulse.duration)
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((target_drive_channel, control_delay))
        sequence.append((ro_channel, control_delay))
        sequence.append((ro_channel_control, control_delay))
        sequence.append((cr_channel, control_delay))

    if echo:
        delays = 6 * [Delay(duration=cr_drive_pulse.duration)]
        cr_pulse_minus = replace(cr_drive_pulse, relative_phase=np.pi)
        cr_pulses.append(cr_pulse_minus)
        sequence.append((cr_channel, cr_drive_pulse))
        sequence.append((control_drive_channel, delays[-1]))
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, Delay(duration=control_drive_pulse.duration)))
        sequence.append((cr_channel, cr_pulse_minus))
        sequence.append((control_drive_channel, delays[-2]))
        sequence.append((control_drive_channel, control_drive_pulse))

    else:
        delays = 3 * [Delay(duration=cr_drive_pulse.duration)]
        sequence.append((cr_channel, cr_drive_pulse))

    if interpolated_sweeper:
        sequence.align(
            [cr_channel, control_drive_channel, ro_channel, ro_pulse_control]
        )
    else:
        sequence.append((ro_channel, delays[0]))
        sequence.append((ro_channel_control, delays[1]))
        sequence.append((target_drive_channel, delays[2]))
        if echo:
            sequence.append((ro_channel, delays[2]))
            sequence.append((ro_channel_control, delays[3]))
            sequence.append(
                (ro_channel, Delay(duration=2 * control_drive_pulse.duration))
            )
            sequence.append(
                (ro_channel_control, Delay(duration=2 * control_drive_pulse.duration))
            )

    if basis == Basis.X:
        # H
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi)))
        sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1],
            )
        )
    elif basis == Basis.Y:
        # SDG
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi / 2)))
        # H
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi)))
        sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1],
            )
        )

    target_delay = Delay(
        duration=natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1].duration
    )
    sequence.append((ro_channel, target_delay))
    sequence.append((ro_channel_control, target_delay))
    sequence.append((ro_channel, ro_pulse))
    sequence.append((ro_channel_control, ro_pulse_control))
    return sequence, cr_pulses, delays
