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

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            if count % points == 0:
                yield data
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
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
    # qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
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
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
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


@store
def rabi_pulse_amplitude(
    platform,
    qubit,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):

    data = Dataset(name=f"data_q{qubit}", quantities={"amplitude": "db"})

    sequence = PulseSequence()
    # qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse = platform.RX_pulse(qubit, start=0)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    count = 0
    for _ in range(software_averages):
        for amplitude in qd_pulse_amplitude_range:
            qd_pulse.amplitude = amplitude
            if count % points == 0:
                yield data
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "amplitude[db]": amplitude,
            }
            data.add(results)
            count += 1
    yield data


@store
def rabi_pulse_length_and_gain(
    platform,
    qubit,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    pulse_gain_start,
    pulse_gain_end,
    pulse_gain_step,
    software_averages,
    points=10,
):

    data = Dataset(name=f"data_q{qubit}", quantities={"duration": "ns", "gain": "db"})

    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            for gain in qd_pulse_gain_range:
                platform.qd_port[qubit].gain = gain
                if count % points == 0:
                    yield data
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "duration[ns]": duration,
                    "gain[db]": gain,
                }
                data.add(results)
                count += 1

    yield data


@store
def rabi_pulse_length_and_amplitude(
    platform,
    qubit,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):

    data = Dataset(
        name=f"data_q{qubit}", quantities={"duration": "ns", "amplitude": "V"}
    )

    sequence = PulseSequence()
    qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    qd_pulse_amplitude_range = np.arange(
        pulse_gain_start, pulse_gain_end, pulse_gain_step
    )

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            for amplitude in qd_pulse_amplitude_range:
                qd_pulse.amplitude = amplitude
                if count % points == 0:
                    yield data
                msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "duration[ns]": duration,
                    "amplitude[V]": amplitude,
                }
                data.add(results)
                count += 1

    yield data
