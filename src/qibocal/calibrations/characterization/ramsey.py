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
    qubit: int,
    t_start,
    t_end,
    t_step,
    n_osc,
    points=10,
):

    r"""
    We introduce an artificial detune over the drive pulse frequency to be off-resonance and, after fitting,
    determine two of the qubit's properties: Ramsey or detuning frequency and T2. If our drive pulse is well
    calibrated, the Ramsey experiment without artificial detuning results in an exponential that describes T2,
    but we can not refine the detuning frequency.

    In this method we iterate over diferent maximum time delays between the drive pulses of the ramsey sequence
    in order to refine the fitted detuning frequency and T2.

    Ramsey sequence: Rx(pi/2) - wait time - Rx(pi/2) - ReadOut

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        t_start (int): Initial time delay between drive pulses in the Ramsey sequence
        t_end (list): List of maximum time delays between drive pulses in the Ramsey sequence
        t_step (int): Scan range step for the time delay between drive pulses in the Ramsey sequence
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Wait time used in the current Ramsey execution
            - **t_max[ns]**: Maximum time delay between drive pulses in the Ramsey sequence

        - A DataUnits object with the fitted data obtained with the following keys

            - **delta_frequency**: Physical detunning of the actual qubit frequency
            - **corrected_qubit_frequency**:
            - **t2**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
    """

    platform.reload_settings()
    sampling_rate = platform.sampling_rate

    data = DataUnits(name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"})

    RX90_pulse1 = platform.create_RX90_pulse(qubit, start=0)
    RX90_pulse2 = platform.create_RX90_pulse(qubit, start=RX90_pulse1.finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse2.finish)

    sequence = PulseSequence()
    sequence.add(RX90_pulse1)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    runcard_qubit_freq = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    runcard_T2 = platform.characterization["single_qubit"][qubit]["T2"]
    intermediate_freq = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "frequency"
    ]

    current_qubit_freq = runcard_qubit_freq
    current_T2 = runcard_T2

    t_end = np.array(t_end)
    for t_max in t_end:
        count = 0
        platform.qd_port[qubit].lo_frequency = current_qubit_freq - intermediate_freq
        offset_freq = n_osc / t_max * sampling_rate  # Hz
        t_range = np.arange(t_start, t_max, t_step)
        for wait in t_range:
            if count % points == 0 and count > 0:
                yield data
                yield ramsey_fit(
                    data,
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubit=qubit,
                    qubit_freq=current_qubit_freq,
                    sampling_rate=sampling_rate,
                    offset_freq=offset_freq,
                    labels=[
                        "delta_frequency",
                        "corrected_qubit_frequency",
                        "t2",
                    ],
                )
            RX90_pulse2.start = RX90_pulse1.finish + wait
            RX90_pulse2.relative_phase = (
                (RX90_pulse2.start / sampling_rate) * (2 * np.pi) * (-offset_freq)
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

        # # Fitting
        data_fit = ramsey_fit(
            data,
            x="wait[ns]",
            y="MSR[uV]",
            qubit=qubit,
            qubit_freq=current_qubit_freq,
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
        if (new_t2 > current_T2) and len(t_end) > 1:
            current_qubit_freq = int(corrected_qubit_freq)
            current_T2 = new_t2
            data = DataUnits(
                name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"}
            )
        else:
            corrected_qubit_freq = int(current_qubit_freq)
            new_t2 = current_T2
            break

    yield data


@plot("MSR vs Time", plots.time_msr)
def ramsey(
    platform: AbstractPlatform,
    qubit: int,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):

    r"""
    The purpose of the Ramsey experiment is to determine two of the qubit's properties: Ramsey or detuning frequency and T2.

    Ramsey sequence: Rx(pi/2) - wait time - Rx(pi/2) - ReadOut

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        delay_between_pulses_start (int): Initial time delay between drive pulses in the Ramsey sequence
        delay_between_pulses_end (list): Maximum time delay between drive pulses in the Ramsey sequence
        delay_between_pulses_step (int): Scan range step for the time delay between drive pulses in the Ramsey sequence
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Wait time used in the current Ramsey execution
            - **t_max[ns]**: Maximum time delay between drive pulses in the Ramsey sequence

        - A DataUnits object with the fitted data obtained with the following keys

            - **delta_frequency**: Physical detunning of the actual qubit frequency
            - **corrected_qubit_frequency**:
            - **t2**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
    """

    platform.reload_settings()
    sampling_rate = platform.sampling_rate
    qubit_freq = platform.characterization["single_qubit"][qubit]["qubit_freq"]

    RX90_pulse1 = platform.create_RX90_pulse(qubit, start=0)
    RX90_pulse2 = platform.create_RX90_pulse(qubit, start=RX90_pulse1.finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse2.finish)

    sequence = PulseSequence()
    sequence.add(RX90_pulse1)
    sequence.add(RX90_pulse2)
    sequence.add(ro_pulse)

    waits = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    data = DataUnits(name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"})
    count = 0
    for _ in range(software_averages):
        for wait in waits:
            if count % points == 0 and count > 0:
                yield data
                yield ramsey_fit(
                    data,
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubit=qubit,
                    qubit_freq=qubit_freq,
                    sampling_rate=sampling_rate,
                    offset_freq=0,
                    labels=[
                        "delta_frequency",
                        "corrected_qubit_frequency",
                        "t2",
                    ],
                )
            RX90_pulse2.start = RX90_pulse1.finish + wait
            ro_pulse.start = RX90_pulse2.finish

            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "wait[ns]": wait,
                "t_max[ns]": delay_between_pulses_end,
            }
            data.add(results)
            count += 1
    yield data
