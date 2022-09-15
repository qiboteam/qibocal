# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency, variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import plot
from qcvv.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__multiplex)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    feedlines,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    software_averages,
    points=10,
):

    platform.reload_settings()
    check_frequency(
        platform, write=True
    )  # FIXME: would not work for multiple feedlines yet
    sequence = PulseSequence()
    nqubits = {}
    for map in platform.qubit_channel_map:
        for feedline in feedlines:
            nqubits[feedline] = []
            if feedline in platform.qubit_channel_map[map]:
                nqubits[feedline] += [int(map)]
                ro_pulse = platform.create_qubit_readout_pulse(int(map), start=0)
                ro_pulse.qubit = int(map)  # @aorgaz repeated no?
                sequence.add(ro_pulse)

    frequency_range = variable_resolution_scanrange(
        lowres_width, lowres_step, highres_width, highres_step
    )

    data = Dataset(
        name=f"data", quantities={"frequency": "Hz", "qubit": "dimensionless"}
    )
    count = 0

    for _ in range(software_averages):
        for freq in frequency_range:
            for feedline in feedlines:
                if count % points == 0:
                    yield data
                freq = platform.ro_port[nqubits[feedline][0]].lo_frequency + freq
                platform.ro_port[nqubits[feedline][0]].lo_frequency = freq

                results = platform.execute_pulse_sequence(sequence)
                for ro_pulse in sequence.ro_pulses:
                    msr, phase, i, q = results[ro_pulse.serial]
                    r = {
                        "MSR[V]": msr,
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "frequency[Hz]": freq + ro_pulse.frequency,
                        "qubit[dimensionless]": np.int64(ro_pulse.qubit),
                    }
                    data.add(r)
                count += 1
    yield data
