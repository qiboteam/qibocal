# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


def qubit_spectroscopy(
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

    data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})
    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start = 0, duration = 5000) 
    ro_pulse = platform.qubit_readout_pulse(qubit, start = 5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    # Fast Sweep
    # if (software_averages !=0):
    #     lo_qcm_frequency = platform.characterization['single_qubit'][qubit]['qubit_freq'] - qd_pulse.frequency
    #     fast_sweep_scan_range = np.arange(fast_start, fast_end, fast_step)
    #     mc.settables(settable(platform.qd_port[qubit], 'lo_frequency', 'Frequency', 'Hz'))
    #     mc.setpoints(fast_sweep_scan_range + lo_qcm_frequency)
    #     mc.gettables(ROController(platform, sequence, qubit))
    #     platform.start() 
    #     dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
    #     platform.stop()

    #     if self.resonator_type == '3D':
    #         lo_qcm_frequency = dataset['x0'].values[dataset['y0'].argmin().values]
    #         avg_voltage = np.mean(dataset['y0']) * 1e6
    #     elif self.resonator_type == '2D':
    #         lo_qcm_frequency = dataset['x0'].values[dataset['y0'].argmax().values]
    #         avg_voltage = np.mean(dataset['y0']) * 1e6

    # # Precision Sweep
    # if (self.software_averages_precision !=0):
    #     precision_sweep_scan_range = np.arange(precision_start, precision_end, precision_step)
    #     mc.settables(settable(platform.qd_port[qubit], 'lo_frequency', 'Frequency', 'Hz'))
    #     mc.setpoints(precision_sweep_scan_range + lo_qcm_frequency)
    #     mc.gettables(ROController(platform, sequence, qubit))
    #     platform.start() 
    #     dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=self.software_averages_precision)
    #     platform.stop()

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

    lo_qrm_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )

    freqrange = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + lo_qrm_frequency
    )
    data = Dataset(name="fast_sweep", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield data
            platform.ro_port[qubit].lo_freq = freq
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[qubit][
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

    if platform.settings["nqubits"] == 1:
        lo_qrm_frequency = data.df.frequency[data.df.MSR.argmax()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6
    else:
        lo_qrm_frequency = data.df.frequency[data.df.MSR.argmin()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6

    prec_data = Dataset(name="precision_sweep", quantities={"frequency": "Hz"})
    freqrange = (
        np.arange(-precision_width, precision_width, precision_step) + lo_qrm_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0:
                yield prec_data
            platform.ro_port[qubit].lo_freq = freq
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[qubit][
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
