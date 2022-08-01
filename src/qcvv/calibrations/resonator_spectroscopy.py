# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def resonator_spectroscopy_attenuation(
    platform,
    qubit,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    min_att,
    max_att,
    step_att,
    software_averages,
):

    data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"}, points=2)
    ro_pulse = platform.qubit_readout_pulse(qubit, 0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)
    freqrange = variable_resolution_scanrange(
        lowres_width,
        lowres_step,
        highres_width,
        highres_step,
    )
    # TODO: move this explicit instruction to the platform
    freqrange = (
        freqrange
        + platform.settings["characterization"]["single_qubit"][qubit]["resonator_freq"]
    )
    attrange = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for s in range(software_averages):
        for freq in freqrange:
            for att in attrange:
                if count % data.points == 0:
                    yield data
                # TODO: move this explicit instruction to the platform
                platform.qrm[qubit].set_device_parameter(
                    "out0_in0_lo_freq", freq + ro_pulse.frequency
                )
                # TODO: move this explicit instruction to the platform
                platform.qrm[qubit].set_device_parameter("out0_att", att)
                msr, i, q, phase = platform.execute_pulse_sequence(sequence, 2000)[
                    qubit
                ][ro_pulse.serial]
                msr, i, q, phase = np.random.rand(4)
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                data.add(results)
                count += 1

    yield data
