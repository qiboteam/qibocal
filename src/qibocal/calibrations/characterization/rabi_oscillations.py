import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import rabi_fit


@plot("MSR vs Time", plots.time_msr_phase)
def rabi_pulse_length(
    platform: AbstractPlatform,
    qubit: int,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(name=f"data_q{qubit}", quantities={"Time": "ns"})

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse.frequency
    )

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            if count % points == 0 and count > 0:
                yield data
                yield rabi_fit(
                    data,
                    x="Time[ns]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=[
                        "pi_pulse_duration",
                        "pi_pulse_max_voltage",
                    ],
                )
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "Time[ns]": duration,
            }
            data.add(results)
            count += 1
    yield data


@plot("MSR vs Gain", plots.gain_msr_phase)
def rabi_pulse_gain(
    platform: AbstractPlatform,
    qubit: int,
    pulse_gain_start,
    pulse_gain_end,
    pulse_gain_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(name=f"data_q{qubit}", quantities={"gain": "dimensionless"})

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.finish)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse.frequency
    )

    count = 0
    for _ in range(software_averages):
        for gain in qd_pulse_gain_range:
            platform.qd_port[qubit].gain = gain
            if count % points == 0 and count > 0:
                yield data
                yield rabi_fit(
                    data,
                    x="gain[dimensionless]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=[
                        "pi_pulse_gain",
                        "pi_pulse_max_voltage",
                    ],
                )
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "gain[dimensionless]": gain,
            }
            data.add(results)
            count += 1
    yield data


@plot("MSR vs Amplitude", plots.amplitude_msr_phase)
def rabi_pulse_amplitude(
    platform,
    qubit: int,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(name=f"data_q{qubit}", quantities={"amplitude": "dimensionless"})

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse.frequency
    )

    count = 0
    for _ in range(software_averages):
        for amplitude in qd_pulse_amplitude_range:
            qd_pulse.amplitude = amplitude
            if count % points == 0 and count > 0:
                yield data
                yield rabi_fit(
                    data,
                    x="amplitude[dimensionless]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=[
                        "pi_pulse_amplitude",
                        "pi_pulse_max_voltage",
                    ],
                )
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "amplitude[dimensionless]": amplitude,
            }
            data.add(results)
            count += 1
    yield data


@plot("MSR vs length and gain", plots.duration_gain_msr_phase)
def rabi_pulse_length_and_gain(
    platform: AbstractPlatform,
    qubit: int,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    pulse_gain_start,
    pulse_gain_end,
    pulse_gain_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"duration": "ns", "gain": "dimensionless"}
    )

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse.frequency
    )

    count = 0
    for _ in range(software_averages):
        for duration in qd_pulse_duration_range:
            qd_pulse.duration = duration
            ro_pulse.start = duration
            for gain in qd_pulse_gain_range:
                platform.qd_port[qubit].gain = gain
                if count % points == 0 and count > 0:
                    yield data
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "duration[ns]": duration,
                    "gain[dimensionless]": gain,
                }
                data.add(results)
                count += 1

    yield data


@plot("MSR vs length and amplitude", plots.duration_amplitude_msr_phase)
def rabi_pulse_length_and_amplitude(
    platform,
    qubit: int,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={"duration": "ns", "amplitude": "dimensionless"},
    )

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse.frequency
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
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "duration[ns]": duration,
                    "amplitude[dimensionless]": amplitude,
                }
                data.add(results)
                count += 1

    yield data
