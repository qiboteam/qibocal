# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency, variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import plot
from qcvv.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
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
    drive_attenuation=None,
    points=10,
):
    platform.reload_settings()
    check_frequency(platform, write=False)

    platform.reload_settings()
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    for i in range(platform.settings["nqubits"]):
        if isinstance(drive_attenuation, list):
            platform.qd_port[i].attenuation = drive_attenuation[i]
        else:
            platform.qd_port[i].attenuation = drive_attenuation

    frequency_range = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + resonator_frequency
    )
    fast_sweep_data = Dataset(
        name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield fast_sweep_data
                yield lorentzian_fit(
                    fast_sweep_data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
            fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data

    # FIXME: have live ploting work for multiple datasets saved

    if platform.resonator_type == "3D":
        resonator_frequency = fast_sweep_data.df.frequency[
            fast_sweep_data.df.MSR.index[fast_sweep_data.df.MSR.argmax()]
        ].magnitude
        avg_voltage = (
            np.mean(fast_sweep_data.df.MSR.values[: (lowres_width // lowres_step)])
            * 1e6
        )
    else:
        resonator_frequency = fast_sweep_data.df.frequency[
            fast_sweep_data.df.MSR.index[fast_sweep_data.df.MSR.argmin()]
        ].magnitude
        avg_voltage = (
            np.mean(fast_sweep_data.df.MSR.values[: (lowres_width // lowres_step)])
            * 1e6
        )

    precision_sweep__data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(-precision_width, precision_width, precision_step)
        + resonator_frequency
    )

    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield precision_sweep__data
                yield lorentzian_fit(
                    fast_sweep_data + precision_sweep__data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
            precision_sweep__data.add(results)
            count += 1
    yield precision_sweep__data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
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
    platform.reload_settings()
    check_frequency(platform, write=False)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].attenuation = att
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr * (np.exp(att / 10)),
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                data.add(results)
                count += 1
    yield data


@plot("MSR and Phase vs Flux Current", plots.frequency_flux_msr_phase)
def resonator_spectroscopy_flux(
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

    if fluxline == "qubit":
        fluxline = qubit

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit][
        "sweetspot"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = (
        np.arange(current_min, current_max, current_step) + qubit_biasing_current
    )

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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
    # TODO: automatically extract the sweet spot current
    # TODO: add a method to generate the matrix


@plot("MSR row 1 and Phase row 2", plots.frequency_flux_msr_phase__matrix)
def resonator_spectroscopy_flux_matrix(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    current_min,
    current_max,
    current_step,
    fluxlines,
    software_averages,
    points=10,
):
    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = np.arange(current_min, current_max, current_step)

    count = 0
    for fluxline in fluxlines:
        fluxline = int(fluxline)
        print(fluxline)
        data = Dataset(
            name=f"data_q{qubit}_f{fluxline}",
            quantities={"frequency": "Hz", "current": "A"},
        )
        for _ in range(software_averages):
            for curr in current_range:
                for freq in frequency_range:
                    if count % points == 0:
                        yield data
                    platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
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


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
def resonator_drive_noise(
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
    platform.reload_settings()
    check_frequency(platform, write=False)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for i in range(platform.nqubits):
                platform.qd_port[qubit].attenuation = att
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                data.add(results)
                count += 1
    yield data
