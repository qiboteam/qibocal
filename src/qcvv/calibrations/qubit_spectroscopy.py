# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence
from qibolab.platforms.abstract import AbstractPlatform
from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def qubit_spectroscopy(
    platform: AbstractPlatform,
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
    platform.reload_settings()
    
    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]

    frequency_range = np.arange(fast_start, fast_end, fast_step) + qubit_frequency

    # FIXME: Waiting for Qblox platform to take care of that
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )

    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data.add(results)
            count += 1
    yield data

    if platform.resonator_type == "3D":
        qubit_frequency = data.df.frequency[data.df.MSR.index[data.df.MSR.argmin()]].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )
    else:
        qubit_frequency = data.df.frequency[data.df.MSR.index[data.df.MSR.argmax()]].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )

    prec_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    frequency_range = (
        np.arange(precision_start, precision_end, precision_step) + qubit_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield prec_data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
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
    platform: AbstractPlatform,
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
    platform.reload_settings()
    

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    frequency_range = np.arange(-freq_width, freq_width, freq_step) + qubit_frequency
    current_range = np.arange(current_min, current_max, current_step)

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qf_port[fluxline].current = curr
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data
