# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def ramsey(
    platform,
    qubit,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):

    sampling_rate = platform.sampling_rate

    RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
    RX90_pulse2 = platform.RX90_pulse(
        qubit,
        start=RX90_pulse1.duration,
        phase=RX90_pulse1.duration / sampling_rate * 2 * np.pi * RX90_pulse1.frequency,
    )
    ro_pulse = platform.qubit_readout_pulse(
        qubit, start=RX90_pulse1.duration + RX90_pulse2.duration
    )

    waits = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"wait": "ns"})
    count = 0
    for _ in range(software_averages):
        for wait in waits:
            if count % points == 0:
                yield data
            RX90_pulse2.start = RX90_pulse1.duration + wait
            ro_pulse.start = RX90_pulse1.duration + wait + RX90_pulse2.duration
            sequence = PulseSequence()
            sequence.add(RX90_pulse1)
            sequence.add(RX90_pulse2)
            sequence.add(ro_pulse)
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "wait[ns]": wait,
            }
            data.add(results)
            count += 1
    yield data

    # # Fitting
    # smooth_dataset, delta_frequency, t2 = fitting.ramsey_fit(dataset)
    # utils.plot(smooth_dataset, dataset, "Ramsey", 1)
    # print(f"\nDelta Frequency = {delta_frequency}")
    # corrected_qubit_frequency = int(
    #     platform.settings["characterization"]["single_qubit"][qubit]["qubit_freq"]
    #     + delta_frequency
    # )
    # print(f"\nCorrected Qubit Frequency = {corrected_qubit_frequency}")
    # print(f"\nT2 = {int(t2)} ns")

    # # TODO: return corrected frequency
    # return (
    #     delta_frequency,
    #     corrected_qubit_frequency,
    #     int(t2),
    #     smooth_dataset,
    #     dataset,
    # )


@store
def run_ramsey_frequency_detuned(self, qubit):
    platform = platform
    platform.reload_settings()
    sampling_rate = platform.sampling_rate
    mc = mc

    reload_settings()
    t_start = settings["ramsey_freq"]["t_start"]
    t_end = settings["ramsey_freq"]["t_end"]
    t_step = settings["ramsey_freq"]["t_step"]
    n_osc = settings["ramsey_freq"]["n_osc"]

    sequence = PulseSequence()
    RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
    RX90_pulse2 = platform.RX90_pulse(
        qubit,
        start=RX90_pulse1.duration,
        phase=(RX90_pulse1.duration / sampling_rate)
        * (2 * np.pi)
        * (RX90_pulse1.frequency),
    )
    ro_pulse = platform.qubit_readout_pulse(
        qubit, start=RX90_pulse1.duration + RX90_pulse2.duration
    )
    sequence.add(RX90_pulse1)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    pl.tuids_max_num(max_num_plots)

    runcard_qubit_freq = platform.settings["characterization"]["single_qubit"][qubit][
        "qubit_freq"
    ]
    runcard_T2 = platform.settings["characterization"]["single_qubit"][qubit]["T2"]
    intermediate_freq = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "frequency"
    ]

    current_qubit_freq = runcard_qubit_freq
    current_T2 = runcard_T2

    for t_max in t_end:
        platform.qd_port[qubit].lo_frequency = current_qubit_freq - intermediate_freq

        offset_freq = n_osc / t_max * sampling_rate  # Hz
        t_range = np.arange(t_start, t_max, t_step)
        mc.settables(
            RamseyFreqWaitParameter(ro_pulse, RX90_pulse2, offset_freq, sampling_rate)
        )
        mc.setpoints(t_range)
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run("Ramsey Frequency Detuned", soft_avg=software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, delta_fitting, new_t2 = fitting.ramsey_freq_fit(dataset)
        delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
        corrected_qubit_freq = int(current_qubit_freq - delta_phys)

        # if ((new_t2 * 3.5) > t_max):
        if new_t2 > current_T2:
            print(
                f"\nFound a better T2: {new_t2}, for a corrected qubit frequency: {corrected_qubit_freq}"
            )
            current_qubit_freq = corrected_qubit_freq
            current_T2 = new_t2
        else:
            print(f"\nCould not find a further improvement on T2")
            corrected_qubit_freq = current_qubit_freq
            new_t2 = current_T2
            break

    return (
        new_t2,
        (corrected_qubit_freq - runcard_qubit_freq),
        corrected_qubit_freq,
        dataset,
    )
