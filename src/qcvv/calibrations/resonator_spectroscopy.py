# -*- coding: utf-8 -*-
import os
import time

import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset


def resonator_spectroscopy_attenuation(platform, qubit, settings, folder):

    path = os.path.join(
        folder, f"resonator_spectroscopy_attenuation/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(path)
    data = Dataset(quantities=[("frequency", "Hz"), ("attenuation", "dB")], points=2)
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
    count = 0
    for s in range(settings["software_average"]):
        for freq in freqrange:
            for att in attrange:
                platform.qrm[qubit].set_device_parameter(
                    "out0_in0_lo_freq", freq + ro_pulse.frequency
                )
                if count % data.points == 0:
                    data.to_yaml(path)
                platform.qrm[qubit].set_device_parameter("out0_att", att)
                res = platform.execute_pulse_sequence(sequence, 2000)[qubit][
                    ro_pulse.serial
                ]
                data.add(*res, [("frequency", "Hz", freq), ("attenuation", "dB", att)])
                count += 1

    data.to_yaml(path)
