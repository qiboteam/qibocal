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

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse duration
    to find the drive pulse length that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **Time[ns]**: Drive pulse duration in ns

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_duration**: pi pulse duration
            - **pi_pulse_max_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
    """

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )

    data = DataUnits(name=f"data_q{qubit}", quantities={"Time": "ns"})

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

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse gain
    to find the drive pulse gain that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_gain_start (int): Initial drive pulse gain for the Rabi experiment
        pulse_gain_end (int): Maximum drive pulse gain for the Rabi experiment
        pulse_gain_step (int): Scan range step for the drive pulse gain for the Rabi experiment
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **gain[dimensionless]**: Drive pulse gain

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_duration**: pi pulse duration
            - **pi_pulse_max_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
    """

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.finish)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    data = DataUnits(name=f"data_q{qubit}", quantities={"gain": "dimensionless"})

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

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **amplitude[dimensionless]**: Drive pulse amplitude

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_amplitude**: pi pulse amplitude
            - **pi_pulse_max_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
    """

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    data = DataUnits(name=f"data_q{qubit}", quantities={"amplitude": "dimensionless"})

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

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse
    combination of duration and gain to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        pulse_gain_start (int): Initial drive pulse gain for the Rabi experiment
        pulse_gain_end (int): Maximum drive pulse gain for the Rabi experiment
        pulse_gain_step (int): Scan range step for the drive pulse gain for the Rabi experiment
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **duration[ns]**: Drive pulse duration in ns
            - **gain[dimensionless]**: Drive pulse gain

    """

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=4)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    qd_pulse_gain_range = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"duration": "ns", "gain": "dimensionless"}
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

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse
    combination of duration and amplitude to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **duration[ns]**: Drive pulse duration in ns
            - **amplitude[dimensionless]**: Drive pulse amplitude

    """

    platform.reload_settings()

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

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={"duration": "ns", "amplitude": "dimensionless"},
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
