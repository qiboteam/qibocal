# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def dispersive_shift(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    software_averages,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )

    data_spec = Dataset(name=f"data_spec_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield data_spec
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_spec.add(results)
            count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    data_shifted = Dataset(
        name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield data_shifted
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_shifted.add(results)
            count += 1
    yield data_shifted
