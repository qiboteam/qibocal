from typing import Optional

import numpy as np
from qibolab import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId


def ramsey_sequence(
    platform: Platform,
    qubit: QubitId,
    wait: Optional[int] = 0,
    detuning: Optional[int] = 0,
):
    """Pulse sequence used in Ramsey experiments."""

    sequence = PulseSequence()
    first_pi_half_pulse = platform.create_RX90_pulse(qubit, start=0)
    second_pi_half_pulse = platform.create_RX90_pulse(
        qubit, start=first_pi_half_pulse.finish + wait
    )
    first_pi_half_pulse.frequency += detuning
    second_pi_half_pulse.frequency += detuning
    readout_pulse = platform.create_qubit_readout_pulse(
        qubit, start=second_pi_half_pulse.finish
    )

    sequence.add(first_pi_half_pulse, second_pi_half_pulse, readout_pulse)
    return sequence


def ramsey_fit(x, offset, amplitude, delta, phase, decay):
    """ "Dumped sinusoidal fit."""
    return offset + amplitude * np.sin(x * delta + phase) * np.exp(-x * decay)
