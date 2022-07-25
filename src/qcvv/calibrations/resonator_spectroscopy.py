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
    for s in range(settings["software_averages"]):
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
    avg_data = data.compute_software_average(["frequency", "attenuation"])
    avg_data.to_yaml(path, name="avg")


def resonator_spectroscopy(platform, qubit, settings, folder):

    path = os.path.join(
        folder, f"resonator_spectroscopy/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(path)
    data = Dataset(quantities=("frequency", "Hz"), points=2)
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
    count = 0
    for s in range(settings["software_averages"]):
        for freq in freqrange:
            if count % data.points == 0:
                data.to_yaml(path, name="sweep")
            platform.qrm[qubit].set_device_parameter(
                "out0_in0_lo_freq", freq + ro_pulse.frequency
            )
            res = platform.execute_pulse_sequence(sequence, 2000)[qubit][
                ro_pulse.serial
            ]
            data.add(*res, ("frequency", "Hz", freq))
            count += 1

    data.to_yaml(path, name="sweep")
    avg_data = data.compute_software_average("frequency")
    avg_data.to_yaml(path, name="sweep_avg")

    platform.qrm[qubit].out0_in0_lo_freq = max(avg_data.get_data("MSR"))
    avg_min_voltage = (
        np.mean(
            avg_data.get_data("MSR")[
                : (settings["lowres_width"] // settings["lowres_step"])
            ]
        )
        * 1e6
    )

    precision_data = Dataset(quantities=("frequency", "Hz"), points=2)
    # Precision sweep
    freqrange = np.arange(
        -settings["precision_width"],
        settings["precision_width"],
        settings["precision_step"],
    )
    freqrange = (
        freqrange
        + platform.settings["characterization"]["single_qubit"][qubit]["resonator_freq"]
    )
    count = 0
    for s in range(settings["software_averages"]):
        for freq in freqrange:
            if count % data.points == 0:
                precision_data.to_yaml(path, name="sweep_precision")
            platform.qrm[qubit].lo.frequency = freq + ro_pulse.frequency
            sequence = PulseSequence()
            sequence.add(ro_pulse)

            res = platform.execute_pulse_sequence(sequence, 1024)[qubit][
                ro_pulse.serial
            ]
            precision_data.add(*res, ("frequency", "Hz", freq))
            count += 1

    precision_data.to_yaml(path, name="sweep_precision")
    precision_avg_data = precision_data.compute_software_average("frequency")
    precision_avg_data.to_yaml(path, name="sweep_precision_avg")

    # Fitting
    from scipy.signal import savgol_filter

    smooth_dataset = savgol_filter(precision_avg_data.get_data("MSR"), 25, 2)
    max_ro_voltage = smooth_dataset.max() * 1e6

    # f0, BW, Q = fitting.lorentzian_fit(
    #     np.array(
    #         [
    #             precision_avg_data.container["frequency"],
    #             precision_avg_data.container["MSR"],
    #         ]
    #     ),
    #     min,
    #     f"Resonator_spectroscopy_qubit{qubit}",
    # )
    # resonator_freq = f0 * 1e9 + ro_pulse.frequency


def resonator_flux_att(platform, qubit, settings, folder):

    path = os.path.join(folder, f"resonator_flux/{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(path)

    ro_pulse = platform.qubit_readout_pulse(qubit, 0)  # start = 0

    # 2D sweep
    currange = np.arange(
        settings["min_current"], settings["max_current"], settings["step_current"]
    )
    spi = platform.instruments["SPI"].device
    dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]

    attrange = np.flip(
        np.arange(settings["min_att"], settings["max_att"], settings["step_att"])
    )

    platform.qrm[qubit].lo.frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        + ro_pulse.frequency
    )

    data = Dataset(quantities=[("current", "A"), ("attenuation", "dB")], points=2)
    count = 0
    for s in range(settings["software_averages"]):
        for att in attrange:
            for curr in currange:
                if count % data.points == 0:
                    data.to_yaml(path, name="sweep")
                dacs[qubit].current(curr)
                platform.qrm[qubit].set_device_parameter("out0_att", att)
                sequence = PulseSequence()
                sequence.add(ro_pulse)

                res = platform.execute_pulse_sequence(sequence, 2000)[qubit][
                    ro_pulse.serial
                ]

                data.add(*res, [("attenuation", "dB", att), ("current", "A", curr)])
                count += 1

    avg_data = data.compute_software_average(["attenuation", "current"])
    avg_data.to_yaml(path, name="sweep_avg")
