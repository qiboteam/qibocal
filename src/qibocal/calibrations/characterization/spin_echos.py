import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import spin_echo_fit


@plot("MSR vs Time", plots.spin_echo_time_msr_phase)
def spin_echo_3pulses(
    platform: AbstractPlatform,
    qubits: list,
    delay_between_pulses_start,
    delay_between_pulses_end,
    delay_between_pulses_step,
    software_averages,
    points=10,
):

    """Spin echo calibration routine:
    The two routines are a modified Ramsey sequence with an additional Rx(pi) pulse placed symmetrically between the two Rx(pi/2) pulses.
    The random phases accumulated before and after the pi pulse compensate exactly if the frequency does not change during the sequence.
    After the additional pi rotation the qubit is expected to go always (ideally) to the ground state. An exponential fit to this data
    gives a spin echo decay time T2.

    Args:
        platfform (AbstrcatPlatform): Qibolab object that allows the user to communicate with the experimental setup (QPU)
        qubit (int): Target qubit to characterize
        delay_between_pulses_start (int): initial delay between pulses
        delay_between_pulses_end (int): end delay between pulses
        delay_between_pulses_step (int): step size for delay between pulses range
        software_averages (int): Number of software repetitions of the experiment
        points (int): Number of points obtained to executed the save method of the results in a file
    """

    platform.reload_settings()

    data = DataUnits(name=f"data", quantities={"time": "ns"}, options=["qubit"])

    ro_pulses = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()

    for qubit in qubits:
        RX90_pulse1 = platform.create_RX90_pulse(qubit, start=0)
        sequence.add(RX90_pulse1)

        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=RX90_pulse1.finish)
        sequence.add(RX_pulses[qubit])

        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        sequence.add(RX90_pulses2[qubit])

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(ro_pulses[qubit])

    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout

    ro_wait_range = np.arange(
        delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step
    )

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0 and count > 0:
                yield data

                for qubit in qubits:
                    yield spin_echo_fit(
                        data.get_column("qubit", qubit),
                        x="time[ns]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["t2"],
                    )

            for qubit in qubits:
                RX_pulses[qubit].start = RX_pulses[qubit].duration + wait
                RX90_pulses2[qubit].start = 2 * RX_pulses[qubit].duration + 2 * wait
                ro_pulses[qubit].start = 3 * RX_pulses[qubit].duration + 2 * wait

            results = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, i, q, phase = results[ro_pulses[qubit].serial]
                result = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[deg]": phase,
                    "time[ns]": wait,
                    "qubit": qubit,
                }
                data.add(result)
            count += 1
    yield data
