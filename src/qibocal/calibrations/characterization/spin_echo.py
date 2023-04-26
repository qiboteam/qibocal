import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import spin_echo_fit


@plot("MSR vs Time", plots.spin_echo_time_msr)
def spin_echo_3pulses(
    platform: AbstractPlatform,
    qubits: list,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages=1,
    points=10,
):
    """Spin echo calibration routine:
    The two routines are a modified Ramsey sequence with an additional Rx(pi) pulse placed symmetrically between the two Rx(pi/2) pulses.
    The random phases accumulated before and after the pi pulse compensate exactly if the frequency does not change during the sequence.
    After the additional pi rotation the qubit is expected to go always (ideally) to the ground state. An exponential fit to this data
    gives a spin echo decay time T2.

    Args:
        platfform (AbstrcatPlatform): Qibolab object that allows the user to communicate with the experimental setup (QPU)
        qubits (list): List of target qubits to perform the action
        delay_between_pulses_start (int): initial delay between pulses
        delay_between_pulses_end (int): end delay between pulses
        delay_between_pulses_step (int): step size for delay between pulses range
        software_averages (int): Number of software repetitions of the experiment
        points (int): Number of points obtained to executed the save method of the results in a file
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    ro_pulses = {}
    RX90_pulses1 = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX_pulses[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step
    )

    data = DataUnits(
        name=f"data", quantities={"wait": "ns"}, options=["qubit", "iteration"]
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for wait in ro_wait_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
                # calculate and save fit
                yield spin_echo_fit(
                    data,
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["t2"],
                )

            for qubit in qubits:
                RX_pulses[qubit].start = RX90_pulses1[qubit].finish + wait
                RX90_pulses2[qubit].start = RX_pulses[qubit].finish + wait
                ro_pulses[qubit].start = RX90_pulses2[qubit].finish

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].raw
                r.update(
                    {
                        "wait[ns]": wait,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
                data.add(r)
            count += 1
    yield data
    yield spin_echo_fit(
        data,
        x="wait[ns]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["t2"],
    )
