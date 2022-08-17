# -*- coding: utf-8 -*-
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
    points=10,

):
    data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})
    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start = 0, duration = 5000) 
    ro_pulse = platform.qubit_readout_pulse(qubit, start = 5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    lo_qcm_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
    )

    freqrange = (
        np.arange(
            fast_start, fast_end, fast_step
        )
        + lo_qcm_frequency
    )

    data = Dataset(name="data", quantities={"frequency": "Hz"})
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

    if platform.settings["nqubits"] == 1:
        lo_qcm_frequency = data.df.frequency[data.df.MSR.argmin()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
    else:
        lo_qcm_frequency = data.df.frequency[data.df.MSR.argmax()].magnitude
        avg_voltage = np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6


    prec_data = Dataset(name="ignore", quantities={"frequency": "Hz"})
    freqrange = (
        np.arange(
            precision_start, precision_end, precision_step
        ) 
        + lo_qcm_frequency
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

    