import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import variable_resolution_scanrange
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    precision_width,
    precision_step,
    software_averages,
    points=10,
):

    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    resonator_frequencies = {}
    frequency_ranges = {}
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        frequency_ranges[qubit] = (
            variable_resolution_scanrange(
                lowres_width, lowres_step, highres_width, highres_step
            )
            + resonator_frequencies[qubit]
        )

    fast_sweep_data = DataUnits(
        name=f"fast_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    )

    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):  # FIXME: remove hardcoding
            if count % points == 0 and count > 0:
                yield fast_sweep_data
                for qubit in qubits:
                    yield lorentzian_fit(
                        fast_sweep_data.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage"],
                    )

            # TODO: move to qibolab platform.set_lo_frequencies(qubits, frequencies)
            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]

                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
# Does not add much value unless one could select the attenuation of the section.
def resonator_punchout(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    min_att,
    max_att,
    step_att,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "attenuation": "dB"},
        options=["qubit"],
    )

    sequence = PulseSequence()
    ro_pulses = {}
    resonator_frequencies = {}
    frequency_ranges = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        frequency_ranges[qubit] = (
            np.arange(-freq_width, freq_width, freq_step)
            + resonator_frequencies[qubit]
            - (freq_width / 4)
        )

    # TODO: move this explicit instruction to the platform

    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for freq in range(len(frequency_ranges[qubit])):
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                for qubit in qubits:
                    platform.ro_port[qubit].lo_frequency = (
                        frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                    )
                    platform.ro_port[qubit].attenuation = att

                result = platform.execute_pulse_sequence(sequence)

                for qubit in qubits:
                    msr, phase, i, q = result[ro_pulses[qubit].serial]
                    results = {
                        "MSR[V]": msr * (np.exp(att / 10)),
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "frequency[Hz]": frequency_ranges[qubit][freq],
                        "attenuation[dB]": att,
                        "qubit": qubit,
                    }
                    # TODO: implement normalization
                    data.add(results)
                count += 1

    yield data


@plot("Flux Dependance", plots.frequency_flux_msr_phase)
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    current_width,
    current_step,
    software_averages,
    fluxlines=None,
    points=10,
):
    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    resonator_frequencies = {}
    delta_frequency_ranges = {}
    sweetspot_currents = {}
    current_ranges = {}
    current_min = {}
    current_max = {}
    ro_pulses = {}

    if not fluxlines:
        fluxlines = qubits

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
    delta_frequency_ranges = np.arange(-freq_width, freq_width, freq_step)

    for fluxline in fluxlines:
        sweetspot_currents[fluxline] = platform.characterization["single_qubit"][qubit][
            "sweetspot"
        ]
        current_min[fluxline] = max(
            -current_width + sweetspot_currents[fluxline], -0.03
        )
        current_max[fluxline] = min(
            +current_width + sweetspot_currents[fluxline], +0.03
        )
        current_ranges[fluxline] = np.arange(
            current_min[fluxline], current_max[fluxline], current_step
        )

    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "current": "A"},
        options=["qubit", "fluxline"],
    )

    count = 0
    for _ in range(software_averages):
        for fluxline in fluxlines:
            for curr in current_ranges[fluxline]:
                platform.qf_port[fluxline].current = curr
                for freq in delta_frequency_ranges:
                    if count % points == 0:
                        yield data

                    for qubit in qubits:
                        platform.ro_port[qubit].lo_frequency = (
                            freq
                            + resonator_frequencies[qubit]
                            - ro_pulses[qubit].frequency
                        )
                    result = platform.execute_pulse_sequence(sequence)

                    for qubit in qubits:
                        msr, phase, i, q = result[ro_pulses[qubit].serial]

                        results = {
                            "MSR[V]": msr,
                            "i[V]": i,
                            "q[V]": q,
                            "phase[rad]": phase,
                            "frequency[Hz]": freq + resonator_frequencies[qubit],
                            "current[A]": curr,
                            "qubit": qubit,
                            "fluxline": fluxline,
                        }
                        data.add(results)
                    count += 1
    yield data


@plot("MSR and Phase vs Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    software_averages,
    points=10,
):

    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    resonator_frequencies = {}
    frequency_ranges = {}
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        frequency_ranges[qubit] = (
            np.arange(-freq_width, freq_width, freq_step) + resonator_frequencies[qubit]
        )

    data_spec = DataUnits(
        name=f"data", quantities={"frequency": "Hz"}, options=["qubit"]
    )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):
            if count % points == 0 and count > 0:
                yield data_spec
                for qubit in qubits:
                    yield lorentzian_fit(
                        data_spec.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage"],
                    )

            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                data_spec.add(results)
                count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        print("ERROR")
        RX_pulse = platform.create_RX_pulse(qubit, start=0)
        print("DONE")
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulse.finish
        )
        sequence.add(RX_pulse)
        sequence.add(ro_pulses[qubit])

    data_shifted = DataUnits(
        name=f"data_shifted", quantities={"frequency": "Hz"}, options=["qubit"]
    )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):
            if count % points == 0 and count > 0:
                yield data_shifted
                for qubit in qubits:
                    yield lorentzian_fit(
                        data_shifted.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage"],
                        fit_file_name="fit_shifted",
                    )
            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                data_shifted.add(results)
                count += 1
    yield data_shifted
