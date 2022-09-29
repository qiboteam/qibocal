# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency, variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import plot
from qcvv.fitting.methods import lorentzian_fit


# @plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubits,
    min_freq,
    max_freq,
    step_freq,
    software_averages,
    points=10,
):

    check_frequency(platform, write=True)
    platform.reload_settings()

    LOs = {}
    sequence = PulseSequence()
    for qubit in qubits:
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulse)
        LOs[qubit] = platform.ro_port[qubit].lo_frequency

    frequency_range = np.arange(min_freq, max_freq, step_freq)
    data = Dataset(name=f"data", quantities={"frequency": "Hz", "qubit": "unit"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            for qubit in qubits:
                if count % points == 0 and count > 0:
                    yield data
                    yield lorentzian_fit(
                        data,
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage"],
                    )
                platform.ro_port[qubit].lo_frequency = freq + LOs[qubit]

            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in sequence.ro_pulses:
                msr, phase, i, q = results[ro_pulse.serial]
                r = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": np.array(
                        platform.ro_port[ro_pulse.qubit].lo_frequency
                        + ro_pulse.frequency
                    ),
                    "qubit[unit]": np.array(ro_pulse.qubit),
                }
                data.add(r)
            count += 1
    yield data


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

    if fluxlines == "diag":
        fluxlines = [qubit]
    elif fluxlines == "all":
        fluxlines = range(platform.nqubits)
    elif fluxlines == "outer":
        fluxlines = np.arange(platform.nqubits)
        fluxlines = fluxlines[fluxlines != qubit]

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
