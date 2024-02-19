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

    sequence = PulseSequence()
    first_pi_half_pulse = platform.create_RX90_pulse(qubit, start=0)
    second_pi_half_pulse = platform.create_RX90_pulse(
        qubit, start=first_pi_half_pulse.finish + wait
    )
    second_pi_half_pulse.frequency += detuning
    readout_pulse = platform.create_qubit_readout_pulse(
        qubit, start=second_pi_half_pulse.finish
    )

    sequence.add(first_pi_half_pulse, second_pi_half_pulse, readout_pulse)
    return sequence


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)
