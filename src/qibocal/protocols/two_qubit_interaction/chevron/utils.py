import numpy as np
from qibolab import PulseSequence, VirtualZ

from qibocal.auto.operation import QubitPairId
from qibocal.calibration import CalibrationPlatform

from ..utils import order_pair

COLORAXIS = ["coloraxis2", "coloraxis1"]

COUPLER_PULSE_START = 0
"""Start of coupler pulse."""
COUPLER_PULSE_DURATION = 100
"""Duration of coupler pulse."""


def chevron_sequence(
    platform: CalibrationPlatform,
    pair: QubitPairId,
    duration_max: int,
    parking: bool = False,
    native: str = "CZ",
    dt: int = 0,
):
    """Chevron pulse sequence."""

    sequence = PulseSequence()
    ordered_pair = order_pair(pair, platform)
    # initialize in system in 11 state
    low_natives = platform.natives.single_qubit[ordered_pair[0]]
    high_natives = platform.natives.single_qubit[ordered_pair[1]]
    if native == "CZ":
        sequence += low_natives.RX()
    sequence += high_natives.RX()

    flux_sequence = getattr(platform.natives.two_qubit[ordered_pair], native)()
    if parking:
        sequence |= [
            (ch, pulse)
            for ch, pulse in flux_sequence
            if not isinstance(pulse, VirtualZ)
        ]
    else:
        target_channels = {platform.qubits[q].flux for q in ordered_pair}
        sequence |= [
            (ch, pulse) for ch, pulse in flux_sequence if ch in target_channels
        ]

    # add readout
    sequence |= low_natives.MZ() + high_natives.MZ()

    return sequence


# fitting function for single row in chevron plot (rabi-like curve)
def chevron_fit(x, omega, phase, amplitude, offset):
    return amplitude * np.cos(x * omega + phase) + offset
