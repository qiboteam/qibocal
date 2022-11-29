import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import variable_resolution_scanrange
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
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

    # avg_voltages = {}
    # freqranges = {}
    # for qubit in qubits:
    #     if platform.resonator_type == "3D":
    #         # resonator_frequencies[qubit] = fast_sweep_data.get_column("qubit", qubit)
    #         resonator_frequencies[qubit] = fast_sweep_data.get_column("qubit", qubit)["frequency"].pint.to("Hz").pint.magnitude[np.argmax(fast_sweep_data.get_column("qubit", qubit)["MSR"].pint.to("V").pint.magnitude)]
    #         avg_voltages[qubit] = np.mean(fast_sweep_data.get_column("qubit", qubit)["MSR"].pint.to("V").pint.magnitude[: (lowres_width // lowres_step)]) * 1e6
    #         print(resonator_frequencies[qubit])
    #         print(avg_voltages[qubit])
    #         # avg_voltages[qubit] = (
    #         #     np.mean(
    #         #         fast_sweep_data.get_values("MSR", "V")[
    #         #             : (lowres_width // lowres_step)
    #         #         ]
    #         #     )
    #         #     * 1e6
    #         # )
    #     else:
    #         # resonator_frequencies[qubit] = fast_sweep_data.get_values(
    #         #     "frequency", "Hz"
    #         # )[np.argmin(fast_sweep_data.get_values("MSR", "V"))]
    #         # avg_voltages[qubit] = (
    #         #     np.mean(
    #         #         fast_sweep_data.get_values("MSR", "V")[
    #         #             : (lowres_width // lowres_step)
    #         #         ]
    #         #     )
    #         #     * 1e6
    #         # )
    #         resonator_frequencies[qubit] = fast_sweep_data.get_column("qubit", qubit)["frequency"].pint.to("Hz").pint.magnitude[np.argmin(fast_sweep_data.get_column("qubit", qubit)["MSR"].pint.to("V").pint.magnitude)]
    #         avg_voltages[qubit] = np.mean(fast_sweep_data.get_column("qubit", qubit)["MSR"].pint.to("V").pint.magnitude[: (lowres_width // lowres_step)]) * 1e6
    #     freqranges[qubit] = np.arange(-precision_width, precision_width, precision_step)
    #     +resonator_frequencies[qubit]

    # precision_sweep__data = DataUnits(
    #     name=f"precision_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    # )

    # count = 0
    # for _ in range(software_averages):
    #     for freq in range(len(freqranges[qubit])):  # FIXME: remove hardcoding
    #         if count % points == 0 and count > 0:
    #             yield precision_sweep__data
    #             for qubit in qubits:
    #                 yield lorentzian_fit(
    #                     (fast_sweep_data + precision_sweep__data).get_column(
    #                         "qubit", qubit
    #                     ),
    #                     x="frequency[GHz]",
    #                     y="MSR[uV]",
    #                     qubit=qubit,
    #                     nqubits=platform.settings["nqubits"],
    #                     labels=["resonator_freq", "peak_voltage"],
    #                 )

    #         for qubit in qubits:
    #             platform.ro_port[qubit].lo_frequency = (
    #                 freqranges[qubit][freq] - ro_pulses[qubit].frequency
    #             )

    #         result = platform.execute_pulse_sequence(sequence)

    #         for qubit in qubits:
    #             msr, phase, i, q = result[ro_pulses[qubit].serial]
    #             results = {
    #                 "MSR[V]": msr,
    #                 "i[V]": i,
    #                 "q[V]": q,
    #                 "phase[rad]": phase,
    #                 "frequency[Hz]": freqranges[qubit][freq],
    #                 "qubit": qubit,
    #             }
    #             precision_sweep__data.add(results)
    #         count += 1
    # yield precision_sweep__data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
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


@plot("MSR and Phase vs Flux Current", plots.frequency_flux_msr_phase)
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit: int,
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

    data = DataUnits(
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
    qubit: int,
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
        data = DataUnits(
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


@plot("MSR and Phase vs Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
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

    data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_spec
                yield lorentzian_fit(
                    data_spec,
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
            data_spec.add(results)
            count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    data_shifted = DataUnits(
        name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_shifted
                yield lorentzian_fit(
                    data_shifted,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                    fit_file_name="fit_shifted",
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
            data_shifted.add(results)
            count += 1
    yield data_shifted
