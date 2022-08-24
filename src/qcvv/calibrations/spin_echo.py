# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.data import Dataset
from qcvv.decorators import store


@store
def spin_echo(
    platform,
    qubit,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):
    sampling_rate = platform.sampling_rate
    sequence = PulseSequence()
    RX90_pulse = platform.RX90_pulse(qubit, start=0)
    RX_pulse = platform.RX_pulse(
        qubit,
        start=RX90_pulse.duration,
        phase=RX90_pulse.duration / sampling_rate * 2 * np.pi * RX90_pulse.frequency,
    )
    ro_pulse = platform.qubit_readout_pulse(
        qubit, start=RX_pulse.start + RX_pulse.duration
    )
    sequence.add(RX90_pulse)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["resonator_freq"] - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["qubit_freq"] - RX_pulse.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0:
                yield data
            RX_pulse.start = RX90_pulse.duration + wait
            RX_pulse.phase = (
                (RX_pulse.start / sampling_rate) * (2 * np.pi) * RX_pulse.frequency
            )
            ro_pulse.start = 2 * RX90_pulse.duration + 2 * wait
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "Time[ns]": wait,
            }
            data.add(results)
            count += 1
    yield data


@store
def spin_echo_3pulses(
    platform,
    qubit,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):
    sampling_rate = platform.sampling_rate
    sequence = PulseSequence()
    RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
    RX_pulse = platform.RX_pulse(qubit, start=RX90_pulse1.duration)
    RX90_pulse2 = platform.RX90_pulse(qubit, start=RX_pulse.start + RX_pulse.duration)
    ro_pulse = platform.qubit_readout_pulse(
        qubit, start=RX90_pulse2.start + RX90_pulse2.duration
    )
    sequence.add(RX90_pulse1)
    sequence.add(RX_pulse)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["resonator_freq"] - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.qpucard["single_qubit"][qubit]["qubit_freq"] - RX_pulse.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0:
                yield data
            RX_pulse.start = RX90_pulse2.duration + wait
            RX_pulse.phase = (
                (RX_pulse.start / sampling_rate) * (2 * np.pi) * RX_pulse.frequency
            )
            RX90_pulse2.start = 2 * RX90_pulse2.duration + 2 * wait
            RX90_pulse2.phase = (
                (RX90_pulse2.start / sampling_rate)
                * (2 * np.pi)
                * RX90_pulse2.frequency
            )
            ro_pulse.start = 3 * RX90_pulse2.duration + 2 * wait
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "Time[ns]": wait,
            }
            data.add(results)
            count += 1
    yield data
