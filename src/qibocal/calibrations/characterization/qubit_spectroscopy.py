import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
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

    qubit_frequencies = {}
    frequency_ranges = {}
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:

        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
        frequency_ranges[qubit] = (
            np.arange(fast_start, fast_end, fast_step) + qubit_frequencies[qubit]
        )

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

        # FIXME: Waiting for Qblox platform to take care of that
        platform.ro_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulses[qubit].frequency
        )

    data = DataUnits(
        name="fast_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    )

    # # FIXME: Waiting for Qblox platform to take care of that
    # platform.ro_port[qubit].lo_frequency = (
    #     platform.characterization["single_qubit"][qubit]["resonator_freq"]
    #     - ro_pulse.frequency
    # )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):  # FIXME: remove hardcoding
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield lorentzian_fit(
                        data.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["qubit_freq", "peak_voltage"],
                    )

            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - qd_pulses[qubit].frequency
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
                data.add(results)
            count += 1
    yield data

    # if platform.resonator_type == "3D":
    #     qubit_frequency = data.get_values("frequency", "Hz")[
    #         np.argmin(data.get_values("MSR", "V"))
    #     ]
    #     avg_voltage = (
    #         np.mean(
    #             data.get_values("MSR", "V")[: ((fast_end - fast_start) // fast_step)]
    #         )
    #         * 1e6
    #     )
    # else:
    #     qubit_frequency = data.get_values("frequency", "Hz")[
    #         np.argmax(data.get_values("MSR", "V"))
    #     ]
    #     avg_voltage = (
    #         np.mean(
    #             data.get_values("MSR", "V")[: ((fast_end - fast_start) // fast_step)]
    #         )
    #         * 1e6
    #     )

    # prec_data = DataUnits(
    #     name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    # )
    # freqrange = (
    #     np.arange(precision_start, precision_end, precision_step) + qubit_frequency
    # )
    # count = 0
    # for _ in range(software_averages):
    #     for freq in freqrange:
    #         if count % points == 0 and count > 0:
    #             yield prec_data
    #             yield lorentzian_fit(
    #                 data + prec_data,
    #                 x="frequency[GHz]",
    #                 y="MSR[uV]",
    #                 qubit=qubit,
    #                 nqubits=platform.settings["nqubits"],
    #                 labels=["qubit_freq", "peak_voltage"],
    #             )
    #         platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
    #         msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
    #             ro_pulse.serial
    #         ]
    #         results = {
    #             "MSR[V]": msr,
    #             "i[V]": i,
    #             "q[V]": q,
    #             "phase[rad]": phase,
    #             "frequency[Hz]": freq,
    #         }
    #         prec_data.add(results)
    #         count += 1
    # yield prec_data
    # TODO: Estimate avg_voltage correctly


@plot("Qubit Flux Dependance", plots.frequency_flux_msr_phase)
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    current_width,
    current_step,
    software_averages,
    fluxlines,
    points=10,
):
    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    qubit_frequencies = {}
    delta_frequency_ranges = {}
    sweetspot_currents = {}
    current_ranges = {}
    current_min = {}
    current_max = {}
    ro_pulses = {}
    qd_pulses = {}

    if fluxlines == "qubits":
        fluxlines = qubits

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=5000
        )
        sequence.add(ro_pulses[qubit])
        sequence.add(qd_pulses[qubit])
        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
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
                        platform.qd_port[qubit].lo_frequency = (
                            freq + qubit_frequencies[qubit] - qd_pulses[qubit].frequency
                        )
                    result = platform.execute_pulse_sequence(sequence)

                    for qubit in qubits:
                        msr, phase, i, q = result[ro_pulses[qubit].serial]

                        results = {
                            "MSR[V]": msr,
                            "i[V]": i,
                            "q[V]": q,
                            "phase[rad]": phase,
                            "frequency[Hz]": freq + qubit_frequencies[qubit],
                            "current[A]": curr,
                            "qubit": qubit,
                            "fluxline": fluxline,
                        }
                        data.add(results)
                    count += 1
    yield data
