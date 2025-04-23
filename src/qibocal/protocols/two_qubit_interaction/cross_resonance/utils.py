from enum import Enum

import numpy as np
from qibolab import Delay, GaussianSquare, Platform, Pulse, PulseSequence, VirtualZ

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
    """Creates sequence for CR experiment on ``control`` and ``target`` qubits.

    With ``setup`` it is possible to set the control qubit to 1 or keep it at 0.
    If ``echo`` is set to ``True`` a ECR gate will be played.
    With ``basis`` it is possible to set the measurement basis. If it is not provided
    the default is Z."""

    cr_pulses = []
    sequence = PulseSequence()
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]
    cr_channel = platform.qubits[control].drive_extra[target]
    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=amplitude,
        relative_phase=0,
        # envelope=Rectangular(),
        envelope=GaussianSquare(rel_sigma=0.2, risefall=15),
    )
    cr_pulses.append(cr_drive_pulse)
    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    target_drive_channel, _ = natives_target.RX()[0]
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
        delays = 8 * [Delay(duration=cr_drive_pulse.duration)]
        control_delay = Delay(duration=control_drive_pulse.duration)
        cr_pulse_minus = replace(cr_drive_pulse, relative_phase=np.pi)
        cr_pulses.append(cr_pulse_minus)
        sequence.append((cr_channel, cr_drive_pulse))
        sequence.append((control_drive_channel, delays[-1]))
        sequence.append((target_drive_channel, delays[-2]))
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, control_delay))
        sequence.append((target_drive_channel, control_delay))
        sequence.append((cr_channel, cr_pulse_minus))
        sequence.append((control_drive_channel, delays[-3]))
        sequence.append((target_drive_channel, delays[-4]))
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((target_drive_channel, control_delay))

    else:
        delays = 3 * [Delay(duration=cr_drive_pulse.duration)]
        sequence.append((cr_channel, cr_drive_pulse))

    if interpolated_sweeper:
        sequence.align(
            [
                cr_channel,
                target_drive_channel,
                control_drive_channel,
                ro_channel,
                ro_channel_control,
            ]
        )
    else:
        sequence.append((ro_channel, delays[0]))
        sequence.append((ro_channel_control, delays[1]))
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
