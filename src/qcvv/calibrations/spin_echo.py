# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def spin_echo(
    platform: AbstractPlatform,
    qubit,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):
    # sampling_rate = platform.sampling_rate

    # Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
    sequence = PulseSequence()
    RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
    RX_pulse = platform.create_RX_pulse(qubit, start=RX90_pulse.finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    sequence.add(RX90_pulse)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - RX_pulse.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0:
                yield data
            RX_pulse.start = RX_pulse.duration + wait
            ro_pulse.start = 2 * RX_pulse.duration + 2 * wait

            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
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
    platform: AbstractPlatform,
    qubit,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):
    # sampling_rate = platform.sampling_rate

    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    sequence = PulseSequence()
    RX90_pulse1 = platform.create_RX90_pulse(qubit, start=0)
    RX_pulse = platform.create_RX_pulse(qubit, start=RX90_pulse1.finish)
    RX90_pulse2 = platform.create_RX90_pulse(qubit, start=RX_pulse.finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse2.finish)

    sequence.add(RX90_pulse1)
    sequence.add(RX_pulse)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - RX_pulse.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0:
                yield data

            RX_pulse.start = RX_pulse.duration + wait
            RX90_pulse2.start = 2 * RX_pulse.duration + 2 * wait
            ro_pulse.start = 3 * RX_pulse.duration + 2 * wait

            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[
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
