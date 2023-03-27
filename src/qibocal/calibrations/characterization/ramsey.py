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
    qubits: dict,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    n_osc,
    software_averages=1,
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
        qubits (dict): Dict of target Qubit objects to perform the action
        delay_between_pulses_start (int): Initial time delay between drive pulses in the Ramsey sequence
        delay_between_pulses_end (list): List of maximum time delays between drive pulses in the Ramsey sequence
        delay_between_pulses_step (int): Scan range step for the time delay between drive pulses in the Ramsey sequence
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Wait time used in the current Ramsey execution
            - **t_max[ns]**: Maximum time delay between drive pulses in the Ramsey sequence
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **delta_frequency**: Physical detunning of the actual qubit frequency
            - **drive_frequency**:
            - **T2**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    # RX90 - wait t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses1[qubit].frequency = qubits[qubit].drive_frequency
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    sampling_rate = platform.sampling_rate

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = DataUnits(
        name="data",
        quantities={"wait": "ns", "t_max": "ns"},
        options=["qubit", "iteration"],
    )

    delay_between_pulses_end = np.array(delay_between_pulses_end)
    # repeat the experiment as many times as defined by software_averages
    count = 0
    for t_max in delay_between_pulses_end:
        for iteration in range(software_averages):
            count = 0
            offset_freq = n_osc / t_max * sampling_rate  # Hz

            # define the parameter to sweep and its range:
            # wait time between RX90 pulses
            t_range = np.arange(
                delay_between_pulses_start, t_max, delay_between_pulses_step
            )

            # sweep the parameter
            for wait in t_range:
                # save data as often as defined by points
                if count % points == 0 and count > 0:
                    # save data
                    yield data
                    # calculate and save fit
                    yield ramsey_fit(
                        data,
                        x="wait[ns]",
                        y="MSR[uV]",
                        qubits=qubits,
                        resonator_type=platform.resonator_type,
                        qubit_freqs={
                            qubit: qubits[qubit].drive_frequency for qubit in qubits
                        },
                        sampling_rate=sampling_rate,
                        offset_freq=offset_freq,
                        labels=[
                            "delta_frequency",
                            "drive_frequency",
                            "T2",
                        ],
                    )

                for qubit in qubits:
                    RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
                    RX90_pulses2[qubit].frequency = qubits[qubit].drive_frequency
                    RX90_pulses2[qubit].relative_phase = (
                        (RX90_pulses2[qubit].start / sampling_rate)
                        * (2 * np.pi)
                        * (-offset_freq)
                    )
                    ro_pulses[qubit].start = RX90_pulses2[qubit].finish

                # execute the pulse sequence
                results = platform.execute_pulse_sequence(sequence)

                for ro_pulse in ro_pulses.values():
                    # average msr, phase, i and q over the number of shots defined in the runcard
                    r = results[ro_pulse.serial].to_dict(average=True)
                    r.update(
                        {
                            "wait[ns]": wait,
                            "t_max[ns]": t_max,
                            "qubit": ro_pulse.qubit,
                            "iteration": iteration,
                        }
                    )
                    data.add(r)
                count += 1

            data_fit = ramsey_fit(
                data,
                x="wait[ns]",
                y="MSR[uV]",
                qubits=qubits,
                resonator_type=platform.resonator_type,
                qubit_freqs={qubit: qubits[qubit].drive_frequency for qubit in qubits},
                sampling_rate=sampling_rate,
                offset_freq=offset_freq,
                labels=[
                    "delta_frequency",
                    "drive_frequency",
                    "T2",
                ],
            )
        stop = False
        for qubit in qubits:
            new_t2 = float(data_fit.df[data_fit.df["qubit"] == qubit]["T2"])
            corrected_qubit_freq = int(
                data_fit.df[data_fit.df["qubit"] == qubit]["drive_frequency"]
            )

            if new_t2 > qubits[qubit].T2 and len(delay_between_pulses_end) > 1:
                print(
                    f"t_max: {t_max} -- new t2: {new_t2} > current t2: {qubits[qubit].T2} new iteration!"
                )
                qubits[qubit].drive_frequency = int(corrected_qubit_freq)
                qubits[qubit].T2 = new_t2
                data = DataUnits(
                    name=f"data",
                    quantities={"wait": "ns", "t_max": "ns"},
                    options=["qubit", "iteration"],
                )
            else:
                print(
                    f"t_max: {t_max} -- new t2: {new_t2} < current t2: {qubits[qubit].T2} stop!"
                )
                # corrected_qubit_freq = int(current_qubit_freqs[qubit])
                # new_t2 = current_T2s[qubit]
                # TODO: These best values need to be saved to the fits file so that they are returned to the user
                stop = True
        if stop:
            break

    yield data
    yield ramsey_fit(
        data,
        x="wait[ns]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        qubit_freqs={qubit: qubits[qubit].drive_frequency for qubit in qubits},
        sampling_rate=sampling_rate,
        offset_freq=0,
        labels=[
            "delta_frequency",
            "drive_frequency",
            "T2",
        ],
    )


@plot("MSR vs Time", plots.time_msr)
def ramsey(
    platform: AbstractPlatform,
    qubits: dict,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages=1,
    points=10,
):
    r"""
    The purpose of the Ramsey experiment is to determine two of the qubit's properties: Ramsey or detuning frequency and T2.

    Ramsey sequence: Rx(pi/2) - wait time - Rx(pi/2) - ReadOut

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        delay_between_pulses_start (int): Initial time delay between drive pulses in the Ramsey sequence
        delay_between_pulses_end (list): Maximum time delay between drive pulses in the Ramsey sequence
        delay_between_pulses_step (int): Scan range step for the time delay between drive pulses in the Ramsey sequence
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Wait time used in the current Ramsey execution
            - **t_max[ns]**: Maximum time delay between drive pulses in the Ramsey sequence
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **delta_frequency**: Physical detunning of the actual qubit frequency
            - **drive_frequency**:
            - **T2**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time between RX90 pulses
    waits = np.arange(
        delay_between_pulses_start,
        delay_between_pulses_end,
        delay_between_pulses_step,
    )

    sampling_rate = platform.sampling_rate

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = DataUnits(
        name=f"data",
        quantities={"wait": "ns", "t_max": "ns"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for wait in waits:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
                # calculate and save fit
                yield ramsey_fit(
                    data,
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    qubit_freqs={
                        qubit: qubits[qubit].drive_frequency for qubit in qubits
                    },
                    sampling_rate=sampling_rate,
                    offset_freq=0,
                    labels=[
                        "delta_frequency",
                        "drive_frequency",
                        "T2",
                    ],
                )

            for qubit in qubits:
                RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
                ro_pulses[qubit].start = RX90_pulses2[qubit].finish

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict(average=True)
                r.update(
                    {
                        "wait[ns]": wait,
                        "t_max[ns]": delay_between_pulses_end,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
                data.add(r)
            count += 1
    yield data
    yield ramsey_fit(
        data,
        x="wait[ns]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        qubit_freqs={qubit: qubits[qubit].drive_frequency for qubit in qubits},
        sampling_rate=sampling_rate,
        offset_freq=0,
        labels=[
            "delta_frequency",
            "drive_frequency",
            "T2",
        ],
    )
