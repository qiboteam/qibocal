
from typing import Optional

import numpy as np
from qibolab import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def ramsey_sequence_mro(
    platform: Platform,
    qubit: QubitId,
    wait: Optional[int] = 0,
    detuning: Optional[int] = 0,
    ro_qubits: Optional[list[QubitId]] = None
):
    """Pulse sequence used in Ramsey (detuned) experiments with multiple RO pulses.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ

    If detuning is specified the RX90 pulses will be sent to
    frequency = drive_frequency + detuning
    """

    if ro_qubits is None:
        ro_qubits = [qubit]

    sequence = PulseSequence()
    first_pi_half_pulse = platform.create_RX90_pulse(qubit, start=0)
    second_pi_half_pulse = platform.create_RX90_pulse(
        qubit, start=first_pi_half_pulse.finish + wait
    )

    # apply detuning:
    first_pi_half_pulse.frequency += detuning
    second_pi_half_pulse.frequency += detuning
    readout_pulse = platform.create_qubit_readout_pulse(
        qubit, start=second_pi_half_pulse.finish
    )

    sequence.add(first_pi_half_pulse, second_pi_half_pulse)

    for ro_qubit in ro_qubits:
        sequence.add(platform.create_qubit_readout_pulse(
            ro_qubit, start=second_pi_half_pulse.finish) )
    return sequence