# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def resonator_spectroscopy(
    platform,
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
    ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    lo_qrm_frequency = platform.qpucard["single_qubit"][qubit]["resonator_freq"]

    freqrange = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + lo_qrm_frequency
    )
    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield data
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
    #     lo_qrm_frequency = data.df.frequency[data.df.MSR.argmax()].magnitude
    #     avg_voltage = np.mean(data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6
    # else:
    #     lo_qrm_frequency = data.df.frequency[data.df.MSR.argmin()].magnitude
    #     avg_voltage = np.mean(data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6

    prec_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(-precision_width, precision_width, precision_step) + lo_qrm_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield prec_data
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
    # TODO: add fitting (possibly without quantify)
    # # Fitting
    # if self.resonator_type == '3D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", max, "Resonator_spectroscopy")
    #     resonator_freq = int(f0 + ro_pulse.frequency)
    # elif self.resonator_type == '2D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", min, "Resonator_spectroscopy")
    #     resonator_freq = int(f0 + ro_pulse.frequency)
    #     # TODO: Fix fitting of minimum values
    # peak_voltage = peak_voltage * 1e6

    # print(f"\nResonator Frequency = {resonator_freq}")
    # return resonator_freq, avg_voltage, peak_voltage, dataset


@store
def resonator_punchout(
    platform,
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
    ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    lo_qrm_frequency = platform.qpucard["single_qubit"][qubit]["resonator_freq"]
    freqrange = np.arange(-freq_width, freq_width, freq_step) + lo_qrm_frequency
    attrange = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for s in range(software_averages):
        for att in attrange:
            for freq in freqrange:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].attenuation = att
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


@store
def resonator_spectroscopy_flux(
    platform,
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
    ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    lo_qrm_frequency = platform.qpucard["single_qubit"][qubit]["resonator_freq"]

    # FIXME: Waitng for abstract platform to have qf_port[qubit] working
    spi = platform.instruments["SPI"].device
    dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]

    scanrange = np.arange(-freq_width, freq_width, freq_step)
    freqs = scanrange + lo_qrm_frequency
    currange = np.arange(current_min, current_max, current_step)

    count = 0
    for s in range(software_averages):
        for curr in currange:
            for freq in freqs:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
    # spi.set_dacs_zero() is this needed?
    # TODO: call platform.qfm[fluxline] instead of dacs[fluxline]
    # TODO: automatically extract the sweet spot current
    # TODO: add a method to generate the matrix
