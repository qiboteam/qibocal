import numpy as np
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId

from ..utils import order_pair

COLORAXIS = ["coloraxis2", "coloraxis1"]

COUPLER_PULSE_START = 0
"""Start of coupler pulse."""
COUPLER_PULSE_DURATION = 100
"""Duration of coupler pulse."""


def chevron_sequence(
    platform: Platform,
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

    if native == "CZ":
        initialize_lowfreq = platform.create_RX_pulse(
            ordered_pair[0], start=0, relative_phase=0
        )
        sequence.add(initialize_lowfreq)

    initialize_highfreq = platform.create_RX_pulse(
        ordered_pair[1], start=0, relative_phase=0
    )
    sequence.add(initialize_highfreq)

    flux_sequence, _ = getattr(platform, f"create_{native}_pulse_sequence")(
        qubits=(ordered_pair[1], ordered_pair[0]),
        start=initialize_highfreq.finish,
    )

    sequence.add(flux_sequence.get_qubit_pulses(ordered_pair[0]))
    sequence.add(flux_sequence.get_qubit_pulses(ordered_pair[1]))

    delay_measurement = duration_max

    if platform.couplers:
        coupler_pulse = flux_sequence.coupler_pulses(
            platform.pairs[tuple(ordered_pair)].coupler.name
        )
        sequence.add(coupler_pulse)
        delay_measurement = max(duration_max, coupler_pulse.duration)

    if parking:
        for pulse in flux_sequence:
            if pulse.qubit not in ordered_pair:
                pulse.start = COUPLER_PULSE_START
                pulse.duration = COUPLER_PULSE_DURATION
                sequence.add(pulse)

    # add readout
    measure_lowfreq = platform.create_qubit_readout_pulse(
        ordered_pair[0],
        start=initialize_highfreq.finish + delay_measurement + dt,
    )
    measure_highfreq = platform.create_qubit_readout_pulse(
        ordered_pair[1],
        start=initialize_highfreq.finish + delay_measurement + dt,
    )

    sequence.add(measure_lowfreq)
    sequence.add(measure_highfreq)

    return sequence


# fitting function for single row in chevron plot (rabi-like curve)
def chevron_fit(x, omega, phase, amplitude, offset):
    return amplitude * np.cos(x * omega + phase) + offset
