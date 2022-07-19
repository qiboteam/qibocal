# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence


def variable_resolution_scanrange(
    lowres_width, lowres_step, highres_width, highres_step
):
    scanrange = np.concatenate(
        (
            np.arange(-lowres_width, -highres_width, lowres_step),
            np.arange(-highres_width, highres_width, highres_step),
            np.arange(highres_width, lowres_width, lowres_step),
        )
    )
    return scanrange


def resonator_spectroscopy(platform, qubit, settings, folder):
    import numpy as np

    ro_pulse = platform.qubit_readout_pulse(qubit, 0)  # start = 0
    sequence = PulseSequence()
    sequence.add(ro_pulse)
    freqrange = variable_resolution_scanrange(
        settings["lowres_width"],
        settings["lowres_step"],
        settings["highres_width"],
        settings["highres_step"],
    )
    freqrange = (
        freqrange
        + platform.settings["characterization"]["single_qubit"][qubit]["resonator_freq"]
    )
    attrange = np.flip(
        np.arange(settings["min_att"], settings["max_att"], settings["step_att"])
    )
    voltages = []
    freqs = []
    powers = []
    for s in range(settings["software_average"]):
        for freq in freqrange:
            for att in attrange:
                platform.qrm[qubit].set_device_parameter(
                    "out0_in0_lo_freq", freq + ro_pulse.frequency
                )
                platform.qrm[qubit].set_device_parameter("out0_att", att)
                res = platform.execute_pulse_sequence(sequence, 2000)
                voltages.append(res[qubit][ro_pulse.serial][0])
                freqs.append(freq)
                powers.append(att)

    np.save(f"{folder}/test.npy", np.array([freqs, voltages, powers]))
