import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import ramsey_fit


@plot("MSR vs Time", plots.time_msr)
def ramsey_frequency_detuned(
    platform: AbstractPlatform,
    qubits: list,
    t_start,
    t_end,
    t_step,
    n_osc,
    software_averages=1,
    points=10,
):
    platform.reload_settings()
    sampling_rate = platform.sampling_rate

    data = DataUnits(
        name=f"data", quantities={"wait": "ns", "t_max": "ns"}, options=["qubit"]
    )

    ro_pulses = {}
    runcard_qubit_freqs = {}
    runcard_T2s = {}
    intermediate_freqs = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    current_qubit_freqs = {}
    current_T2s = {}

    sequence = PulseSequence()
    for qubit in qubits:

        RX90_pulses1["qubit"] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2["qubit"] = platform.create_RX90_pulse(
            qubit, start=RX90_pulses1["qubit"].finish
        )
        ro_pulses["qubit"] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2["qubit"].finish
        )

        sequence.add(RX90_pulses1["qubit"])
        sequence.add(RX90_pulses2["qubit"])
        sequence.add(ro_pulses["qubit"])

        runcard_qubit_freqs["qubit"] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
        runcard_T2s["qubit"] = platform.characterization["single_qubit"][qubit]["T2"]
        intermediate_freqs["qubit"] = platform.settings["native_gates"]["single_qubit"][
            qubit
        ]["RX"]["frequency"]

        # TODO: fix this
        current_qubit_freqs["qubit"] = runcard_qubit_freqs["qubit"]
        current_T2s["qubit"] = runcard_T2s["qubit"]

        # FIXME: Waiting to be able to pass qpucard to qibolab
        platform.ro_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulses["qubit"].frequency
        )
        platform.qd_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["qubit_freq"]
            - RX90_pulses1["qubit"].frequency
        )

    t_end = np.array(t_end)
    for _ in range(software_averages):
        for t_max in t_end:
            count = 0
            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    current_qubit_freqs["qubit"] - intermediate_freqs["qubit"]
                )
            offset_freq = n_osc / t_max * sampling_rate  # Hz
            t_range = np.arange(t_start, t_max, t_step)
            for wait in t_range:
                if count % points == 0 and count > 0:
                    yield data
                    for qubit in qubits:
                        yield ramsey_fit(
                            data.get_column("qubit", qubit),
                            x="wait[ns]",
                            y="MSR[uV]",
                            qubit=qubit,
                            qubit_freq=current_qubit_freqs["qubit"],
                            sampling_rate=sampling_rate,
                            offset_freq=offset_freq,
                            labels=[
                                "delta_frequency",
                                "corrected_qubit_frequency",
                                "t2",
                            ],
                        )

                for qubit in qubits:
                    RX90_pulses2["qubit"].start = RX90_pulses1["qubit"].finish + wait
                    RX90_pulses2["qubit"].relative_phase = (
                        (RX90_pulses2["qubit"].start / sampling_rate)
                        * (2 * np.pi)
                        * (-offset_freq)
                    )
                    ro_pulses["start"].start = RX90_pulses2["start"].finish

                result = platform.execute_pulse_sequence(sequence)

                for qubit in qubits:
                    msr, phase, i, q = result[ro_pulses["qubit"].serial]
                    results = {
                        "MSR[V]": msr,
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "wait[ns]": wait,
                        "t_max[ns]": t_max,
                        "qubit": qubit,
                    }
                    data.add(results)
                # count += 1

                # # Fitting
                data_fit = ramsey_fit(
                    data.get_column("qubit", qubit),
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubit=qubit,
                    qubit_freq=current_qubit_freqs["qubit"],
                    sampling_rate=sampling_rate,
                    offset_freq=offset_freq,
                    labels=[
                        "delta_frequency",
                        "corrected_qubit_frequency",
                        "t2",
                    ],
                )

                new_t2 = data_fit.get_values("t2")
                corrected_qubit_freq = data_fit.get_values("corrected_qubit_frequency")

                # if ((new_t2 * 3.5) > t_max):
                if (new_t2 > current_T2s["qubit"]).all() and len(t_end) > 1:
                    current_qubit_freqs["qubit"] = int(corrected_qubit_freq)
                    current_T2s["qubit"] = new_t2
                    data = DataUnits(
                        name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"}
                    )
                else:
                    corrected_qubit_freq = int(current_qubit_freqs["qubit"])
                    new_t2 = current_T2s["qubit"]
                    break
            count += 1
    yield data


@plot("MSR vs Time", plots.time_msr)
def ramsey(
    platform: AbstractPlatform,
    qubits: list,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):
    platform.reload_settings()
    sampling_rate = platform.sampling_rate

    qubit_freqs = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:

        qubit_freqs["qubit"] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
        RX90_pulses1["qubit"] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2["qubit"] = platform.create_RX90_pulse(
            qubit, start=RX90_pulses1["qubit"].finish
        )
        ro_pulses["qubit"] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2["qubit"].finish
        )

        sequence.add(RX90_pulses1["qubit"])
        sequence.add(RX90_pulses2["qubit"])
        sequence.add(ro_pulses["qubit"])

        # FIXME: Waiting to be able to pass qpucard to qibolab
        platform.ro_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulses["qubit"].frequency
        )
        platform.qd_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["qubit_freq"]
            - RX90_pulses1["qubit"].frequency
        )

    waits = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    data = DataUnits(
        name=f"data", quantities={"wait": "ns", "t_max": "ns"}, options=["qubit"]
    )
    count = 0
    for _ in range(software_averages):
        for wait in waits:
            if count % points == 0 and count > 0:
                yield data

                for qubit in qubits:
                    yield ramsey_fit(
                        data.get_column("qubit", qubit),
                        x="wait[ns]",
                        y="MSR[uV]",
                        qubit=qubit,
                        qubit_freq=qubit_freqs["qubit"],
                        sampling_rate=sampling_rate,
                        offset_freq=0,
                        labels=[
                            "delta_frequency",
                            "corrected_qubit_frequency",
                            "t2",
                        ],
                    )

            for qubit in qubits:
                RX90_pulses2["qubit"].start = RX90_pulses1["qubit"].finish + wait
                ro_pulses["qubit"].start = RX90_pulses2["qubit"].finish

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:

                msr, phase, i, q = result[ro_pulses["qubit"].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "wait[ns]": wait,
                    "t_max[ns]": delay_between_pulses_end,
                    "qubit": qubit,
                }
                data.add(results)
            count += 1
    yield data
