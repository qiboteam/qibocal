# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.data import Dataset
from qcvv.decorators import store

@store
def rabi_pulse_length(
    platform,
    qubit,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    software_averages,
    points=10,
):

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(pulse_duration_start, pulse_duration_end, pulse_duration_step)

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            if count % points == 0:
                yield data
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][ro_pulse.serial]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "Time[ns]": duration,
            }
            data.add(results)
            count += 1
    yield data

@store
def rabi_pulse_gain(
    platform,
    qubit,
    pulse_gain_start,
    pulse_gain_end,
    pulse_gain_step,
    software_averages,
    points=10,
):

    data = Dataset(name=f"data_q{qubit}", quantities={"gain": "db"})

    sequence = PulseSequence()
    #qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse = platform.RX_pulse(qubit, start=0)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    count = 0
    for _ in range(software_averages):
        for gain in qd_pulse_gain_range:
            platform.qd_port[qubit].gain = gain
            if count % points == 0:
                yield data
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][ro_pulse.serial]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "gain[db]": gain,
            }
            data.add(results)
            count += 1
    yield data