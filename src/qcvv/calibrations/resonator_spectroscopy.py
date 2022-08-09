# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence
from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store

@store
def resonator_spectroscopy(platform, qubit, lowres_width, lowres_step, highres_width, highres_step, precision_width, precision_step,software_averages,
    points=10):

    sequence = PulseSequence()
    ro_pulse = platform.qubit_readout_pulse(qubit, start = 0)
    sequence.add(ro_pulse)

    lo_qrm_frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency

    freqrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step) + lo_qrm_frequency
    data = Dataset(name = 'fast_sweep', quantities={'frequency' : 'Hz'})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                    yield data
            platform.ro_port[qubit].lo_freq = freq 
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
                    qubit
                ][ro_pulse.serial]
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

    if platform.settings['nqubits'] == 1:
        lo_qrm_frequency = data.df.frequency[data.df.MSR.argmax()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[:(lowres_width//lowres_step)]).magnitude * 1e6
    else:
        lo_qrm_frequency = data.df.frequency[data.df.MSR.argmin()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[:(lowres_width//lowres_step)]).magnitude * 1e6

    prec_data = Dataset(name = 'precision_sweep', quantities={'frequency' : 'Hz'})
    freqrange = np.arange(-precision_width, precision_width, precision_step)
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                    yield prec_data
            platform.ro_port[qubit].lo_freq = freq 
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
                    qubit
                ][ro_pulse.serial]
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
    #TODO: add fitting (possibly without quantify)
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
    points=10,
):

    data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})
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
                if count % points == 0:
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

# @store
# def resonator_flux(
#     platform,
#     qubit,
#     min_freq,
#     max_freq,
#     step_freq,
#     current_offset,
#     step_current,
#     flux,
#     software_averages,
#     points=10,
# ):

#     data = Dataset(quantities={"frequency": "Hz", "current": "A"})
#     drive_pulse = platform.qubit_drive_pulse(qubit=qubit, start=0)
#     ro_pulse = platform.qubit_readout_pulse(qubit=qubit, start=drive_pulse.duration)
#     sequence = PulseSequence()
#     sequence.add(ro_pulse)
#     sequence.add(drive_pulse)

#     # TODO: move this explicit instruction to the platform
#     spi = platform.instruments["SPI"].device
#     dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]
   
#     start = platform.settings["characterization"]["single_qubit"][qubit]["sweetspot"]
#     currange = np.arange(-current_offset, current_offset, step_current) + start
#     freqrange = np.arange(min_freq, max_freq, step_freq) + platform.settings["characterization"]["single_qubit"][qubit]["qubit_freq"]

#     for s in range(software_averages):
#         for curr in currange:
#             for freq in freqrange:
#                 if count % points == 0:
#                     yield data
#                 platform.qcm[qubit].set_device_parameter(f"out0_lo_freq", freq - drive_pulse.frequency)   
#                 dacs[flux].current(curr)

#                 msr, i, q, phase = platform.execute_pulse_sequence(sequence, 2000)[
#                     qubit
#                 ][ro_pulse.serial]
#                 results = {
#                     "MSR[V]": msr,
#                     "i[V]": i,
#                     "q[V]": q,
#                     "phase[deg]": phase,
#                     "frequency[Hz]": freq,
#                     "current[A]": curr,
#                 }
#                 data.add(results)
#                 count += 1
#     yield data

