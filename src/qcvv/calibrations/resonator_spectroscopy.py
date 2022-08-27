# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence
from qibolab.platforms.abstract import AbstractPlatform
from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubit,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    precision_width,
    precision_step,
    software_averages,
    points=10,
):

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + resonator_frequency
    )
    fast_sweep_data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield fast_sweep_data
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "frequency[Hz]": freq,
            }
            fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data

    # FIXME: have live ploting work for multiple datasets saved

    if platform.resonator_type == "3D":
        resonator_frequency = fast_sweep_data.df.frequency[fast_sweep_data.df.MSR.index[fast_sweep_data.df.MSR.argmax()]].magnitude
        avg_voltage = np.mean(fast_sweep_data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6
    else:
        resonator_frequency = fast_sweep_data.df.frequency[fast_sweep_data.df.MSR.index[fast_sweep_data.df.MSR.argmin()]].magnitude
        avg_voltage = np.mean(fast_sweep_data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6

    precision_sweep_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    frequency_range = (
        np.arange(-precision_width, precision_width, precision_step) + resonator_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield precision_sweep_data
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "frequency[Hz]": freq,
            }
            precision_sweep_data.add(results)
            count += 1
    yield precision_sweep_data

    # TODO: add fitting (possibly without quantify)
    # # Fitting
    # if platform.resonator_type == '3D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", max, "Resonator_spectroscopy")
    #     resonator_freq = int(f0 + ro_pulse.frequency)
    # elif platform.resonator_type == '2D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", min, "Resonator_spectroscopy")
    #     resonator_freq = int(f0 + ro_pulse.frequency)
    #     # TODO: Fix fitting of minimum values
    # peak_voltage = peak_voltage * 1e6

    # print(f"\nResonator Frequency = {resonator_freq}")
    # return resonator_freq, avg_voltage, peak_voltage, dataset


@store
def resonator_punchout(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    min_att,
    max_att,
    step_att,
    software_averages,
    points=10,
):

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    print(type(qubit))
    for _ in range(software_averages):
        for att in attenuation_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].attenuation = att
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "Normalised_MSR[V]": msr * (np.exp(att / 10)),
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                data.add(results)
                count += 1
    yield data


@store
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    current_min,
    current_max,
    current_step,
    software_averages,
    fluxline=0,
    points=10,
):

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    lo_qrm_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    # FIXME: Waitng for abstract platform to have qf_port[qubit] working
    spi = platform.instruments["SPI"].device
    dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]

    scanrange = np.arange(-freq_width, freq_width, freq_step)
    freqs = scanrange + lo_qrm_frequency
    currange = np.arange(current_min, current_max, current_step)

    count = 0
    for s in range(software_averages):
        for freq in freqs:
            for curr in currange:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                # platform.qf_port[fluxline].current = curr
                dacs[fluxline].current(curr)
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
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
    # spi.set_dacs_zero() is this needed?
    # TODO: call platform.qfm[fluxline] instead of dacs[fluxline]
    # TODO: automatically extract the sweet spot current
    # TODO: add a method to generate the matrix
