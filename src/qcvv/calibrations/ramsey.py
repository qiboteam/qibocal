# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence, Pulse
from qibolab.platforms.abstract import AbstractPlatform
from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def ramsey_frequency_detuned(
    platform: AbstractPlatform, qubit, t_start, t_end, t_step, n_osc, software_averages, points=10
):
    platform.reload_settings()

    sampling_rate = platform.sampling_rate

    RX90_pulse1 = platform.create_RX90_pulse(qubit, start=0)
    RX90_pulse2 = platform.create_RX90_pulse(qubit,start=RX90_pulse1.finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse2.finish)

    sequence = PulseSequence()
    sequence.add(RX90_pulse1)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    runcard_qubit_freq = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    runcard_T2 = platform.characterization["single_qubit"][qubit]["T2"]
    intermediate_freq = platform.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]

    current_qubit_freq = runcard_qubit_freq
    current_T2 = runcard_T2

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - RX90_pulse1.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"})
    count = 0
    t_end = np.array(t_end)
    for _ in range(software_averages):
        for t_max in t_end:
            platform.qd_port[qubit].lo_frequency = (
                current_qubit_freq - intermediate_freq
            )
            offset_freq = n_osc / t_max * sampling_rate  # Hz
            t_range = np.arange(t_start, t_max, t_step)
            for wait in t_range:
                if count % points == 0:
                    yield data

                RX90_pulse2.start = RX90_pulse1.finish + wait
                RX90_pulse2.relative_phase = (
                    (RX90_pulse2.start / sampling_rate)
                    * (2 * np.pi)
                    * (-offset_freq)
                )
                ro_pulse.start = RX90_pulse2.finish

                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "wait[ns]": wait,
                    "t_max[ns]": t_max,
                }
                data.add(results)
                count += 1

            # # # Fitting
            # smooth_dataset, delta_fitting, new_t2 = fitting.ramsey_freq_fit(dataset)
            # delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
            # corrected_qubit_freq = int(current_qubit_freq - delta_phys)

            # # if ((new_t2 * 3.5) > t_max):
            # if new_t2 > current_T2:
            #     print(
            #         f"\nFound a better T2: {new_t2}, for a corrected qubit frequency: {corrected_qubit_freq}"
            #     )
            #     current_qubit_freq = corrected_qubit_freq
            #     current_T2 = new_t2
            # else:
            #     print(f"\nCould not find a further improvement on T2")
            #     corrected_qubit_freq = current_qubit_freq
            #     new_t2 = current_T2
            #     break

    yield data
