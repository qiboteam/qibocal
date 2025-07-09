from typing import Optional

import numpy as np
from qibolab import Delay, PulseSequence, VirtualZ

from qibocal.auto.operation import QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

COLORAXIS = ["coloraxis2", "coloraxis1"]

COUPLER_PULSE_START = 0
"""Start of coupler pulse."""
COUPLER_PULSE_DURATION = 100
"""Duration of coupler pulse."""


def chevron_sequence(
    platform: CalibrationPlatform,
    ordered_pair: QubitPairId,
    duration_max: Optional[int] = None,
    parking: bool = False,
    native: str = "CZ",
    dt: int = 0,
):
    """Chevron pulse sequence."""
    sequence = PulseSequence()
    low_natives = platform.natives.single_qubit[ordered_pair[0]]
    high_natives = platform.natives.single_qubit[ordered_pair[1]]
    if native == "CZ":
        sequence += low_natives.RX()
    sequence += high_natives.RX()

    drive_duration = sequence.duration
    raw_flux_sequence = getattr(platform.natives.two_qubit[ordered_pair], native)()
    flux_channel, flux_pulse = [
        (ch, pulse)
        for ch, pulse in raw_flux_sequence
        if ch == platform.qubits[ordered_pair[1]].flux
    ][0]

    if duration_max is not None:
        flux_pulse = replace(flux_pulse, duration=duration_max)

    sequence.append((flux_channel, Delay(duration=drive_duration)))
    sequence.append((flux_channel, flux_pulse))

    parking_pulses = []
    if parking:
        for ch, pulse in raw_flux_sequence:
            if not isinstance(pulse, VirtualZ) and ch != flux_channel:
                sequence.append((ch, Delay(duration=drive_duration)))
                sequence.append((ch, pulse))
                parking_pulses.append(pulse)

    flux_duration = max(flux_pulse.duration, raw_flux_sequence.duration)

    ro_low_channel, ro_high_channel = (
        platform.qubits[ordered_pair[0]].acquisition,
        platform.qubits[ordered_pair[1]].acquisition,
    )
    ro_low_delay = ro_high_delay = drive_delay = Delay(duration=flux_duration)
    dt_delay = Delay(duration=dt)
    drive_channel, second_rx = high_natives.RX()[0]
    sequence += [
        (ro_low_channel, Delay(duration=drive_duration)),
        (ro_high_channel, Delay(duration=drive_duration)),
        (ro_low_channel, ro_low_delay),
        (ro_high_channel, ro_high_delay),
        (ro_low_channel, dt_delay),
        (ro_high_channel, dt_delay),
        (drive_channel, drive_delay),
        (drive_channel, dt_delay),
    ]

    if native == "CZ":
        sequence += [
            (ro_low_channel, Delay(duration=second_rx.duration)),
            (ro_high_channel, Delay(duration=second_rx.duration)),
            (drive_channel, second_rx),
        ]

    # add readout
    sequence += low_natives.MZ() + high_natives.MZ()

    return (
        sequence,
        flux_pulse,
        parking_pulses,
        [ro_low_delay, ro_high_delay, drive_delay],
    )


# fitting function for single row in chevron plot (rabi-like curve)
def chevron_fit(x, omega, phase, amplitude, offset):
    return amplitude * np.cos(x * omega + phase) + offset
