# -*- coding: utf-8 -*-
import importlib
import re

import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def qubit_spectroscopy(
    platform,
    qubit,
    fast_start,
    fast_end,
    fast_step,
    precision_start,
    precision_end,
    precision_step,
    software_averages,
    attenuation,
    points=10,
):
    # data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})
    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    lo_qcm_frequency = platform.qpucard["single_qubit"][qubit]["qubit_freq"]

    freqrange = np.arange(fast_start, fast_end, fast_step) + lo_qcm_frequency

    # FIXME: Waiting for Qblox platform to take care of that
    platform.ro_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["resonator_freq"] - ro_pulse.frequency
    )
    for i in range(platform.settings["nqubits"]):
        if isinstance(attenuation, list):
            platform.qd_port[i].attenuation = attenuation[i]
        else:
            platform.qd_port[i].attenuation = attenuation

    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "frequency[Hz]": freq,
            }
            data.add(results)
            count += 1
    yield data

    # if platform.settings["nqubits"] == 1:
    #     lo_qcm_frequency = data.df.frequency[data.df.MSR.argmin()].magnitude
    #     avg_voltage = (
    #         np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
    #     )
    # else:
    #     lo_qcm_frequency = data.df.frequency[data.df.MSR.argmax()].magnitude
    #     avg_voltage = (
    #         np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
    #     )

    prec_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(precision_start, precision_end, precision_step) + lo_qcm_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield prec_data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "frequency[Hz]": freq,
            }
            prec_data.add(results)
            count += 1
    yield prec_data

    # Fitting
    # if self.resonator_type == '3D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", min, "Qubit_spectroscopy")
    #     qubit_freq = int(f0 + qd_pulse.frequency)
    #     # TODO: Fix fitting of minimum values
    # elif self.resonator_type == '2D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", max, "Qubit_spectroscopy")
    #     qubit_freq = int(f0 + qd_pulse.frequency)

    # # TODO: Estimate avg_voltage correctly
    # print(f"\nQubit Frequency = {qubit_freq}")
    # return qubit_freq, avg_voltage, peak_voltage, dataset


@store
def qubit_spectroscopy_flux(
    platform,
    qubit,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    fluxline=0,
    points=10,
):

    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    # FIXME: Waitng for abstract platform to have qf_port[qubit] working
    spi = platform.instruments["SPI"].device
    dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    lo_qcm_frequency = platform.qpucard["single_qubit"][qubit]["qubit_freq"]
    freqrange = np.arange(-freq_width, freq_width, freq_step) + lo_qcm_frequency
    currange = np.arange(current_min, current_max, current_step)

    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            for curr in currange:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = (
                    np.poly1d(
                        platform.qpuruncard["single_qubit"][qubit][
                            "resonator_polycoef_flux"
                        ]
                    )(curr)
                    - ro_pulse.frequency
                )
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                # platform.qf_port[fluxline].current = curr
                dacs[fluxline].current(curr)
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


@store
def qubit_attenuation(
    platform,
    qubit,
    freq_start,
    freq_end,
    freq_step,
    attenuation_list,
    software_averages,
    points=10,
):

    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["resonator_freq"] - ro_pulse.frequency
    )

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )

    lo_qcm_frequency = platform.qpucard["single_qubit"][qubit]["qubit_freq"]
    freqrange = np.arange(freq_start, freq_end, freq_step) + lo_qcm_frequency

    if isinstance(attenuation_list, str):
        attenuation_list = eval(attenuation_list)

    count = 0
    attenuation_list = np.array(attenuation_list)
    for _ in range(software_averages):
        for att in attenuation_list:
            for freq in freqrange:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qd_port[qubit].attenuation = att
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data
