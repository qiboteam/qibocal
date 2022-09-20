# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency
from qcvv.data import Dataset
from qcvv.decorators import plot
from qcvv.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubit,
    fast_start,
    fast_end,
    fast_step,
    precision_start,
    precision_end,
    precision_step,
    attenuation,
    software_averages,
    points=10,
):
    platform.reload_settings()
    check_frequency(platform, write=True)

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse.frequency = 1.0e6
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    platform.qd_port[qubit].attenuation = attenuation

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]

    freqrange = np.arange(fast_start, fast_end, fast_step) + qubit_frequency

    data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})

    # FIXME: Waiting for Qblox platform to take care of that
    # platform.ro_port[qubit].lo_frequency = (
    #     platform.characterization["single_qubit"][qubit]["resonator_freq"]
    #     - ro_pulse.frequency
    # )

    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield data
                yield lorentzian_fit(
                    data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["qubit_freq", "peak_voltage"],
                )

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
        qubit_frequency = data.df.frequency[
            data.df.MSR.index[data.df.MSR.argmin()]
        ].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )
    else:
        qubit_frequency = data.df.frequency[
            data.df.MSR.index[data.df.MSR.argmax()]
        ].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )

    prec_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(precision_start, precision_end, precision_step) + qubit_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield prec_data
                yield lorentzian_fit(
                    data + prec_data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["qubit_freq", "peak_voltage"],
                )
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
    # TODO: Estimate avg_voltage correctly


@plot("MSR and Phase vs Frequency", plots.frequency_flux_msr_phase)
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    attenuation,
    fluxline,
    points=10,
):
    platform.reload_settings()
    check_frequency(platform, write=True)
    platform.reload_settings()

    if fluxline == "qubit":
        fluxline = qubit

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse.frequency = 1.0e6
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    platform.qd_port[qubit].attenuation = attenuation

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    frequency_range = np.arange(-freq_width, freq_width, freq_step) + qubit_frequency
    current_range = np.arange(current_min, current_max, current_step)

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            platform.ro_port[qubit].lo_frequency = (
                np.poly1d(
                    platform.characterization["single_qubit"][qubit][
                        "resonator_polycoef_flux"
                    ]
                )(curr)
                - ro_pulse.frequency
            )
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qf_port[qubit].current = curr
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


@plot("MSR (row 1) and Phase (row 2)", plots.frequency_flux_msr_phase)
def qubit_spectroscopy_flux_track(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    current_offset,
    current_step,
    software_averages,
    attenuation=46,
    points=10,
):
    platform.reload_settings()
    check_frequency(platform, write=True)

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse.frequency = 1.0e6
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    platform.qd_port[qubit].attenuation = attenuation

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    frequency_array = np.arange(-freq_width, freq_width, freq_step)
    sweetspot = platform.characterization["single_qubit"][qubit]["sweetspot"]
    current_range = np.arange(0, current_offset, current_step)
    current_range = np.append(current_range, -current_range) + sweetspot

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            platform.ro_port[qubit].lo_frequency = (
                np.poly1d(
                    platform.characterization["single_qubit"][qubit][
                        "resonator_polycoef_flux"
                    ]
                )(curr)
                - ro_pulse.frequency
            )

            if curr == sweetspot:
                center = qubit_frequency
                msrs = []
            else:
                idx = np.argmax(msrs)
                center = np.mean(frequency_range[idx])
                msrs = []

            frequency_range = frequency_array + center
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qf_port[qubit].current = curr
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
                msrs += [msr]
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
def qubit_attenuation(
    platform: AbstractPlatform,
    qubit,
    freq_start,
    freq_end,
    freq_step,
    attenuation_list,
    software_averages,
    points=10,
):
    platform.reload_settings()
    check_frequency(platform, write=True)

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse.frequency = 1.0e6
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )

    lo_qcm_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
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
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
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
