import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import flipping_fit


# @plot("MSR vs Flips", plots.flips_msr_phase)
def flipping(
    platform: AbstractPlatform,
    qubits: list,
    niter,
    step,
    software_averages=1,
    points=10,
):

    r"""
    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        niter (int): Maximum number of flips introduced in each sequence
        step (int): Scan range step for the number of flippings
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **flips[dimensionless]**: Number of flips applied in the current execution

        - A DataUnits object with the fitted data obtained with the following keys

            - **amplitude_delta**: Pi pulse delta amplitude
            - **corrected_amplitude**: Corrected pi pulse amplitude
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **popt3**: p3
    """

    platform.reload_settings()

    data = DataUnits(
        name="data",
        quantities={"flips": "dimensionless"},
        options=["qubit", "iteration"],
    )
    pi_pulse_amplitudes = {}
    for qubit in qubits:
        pi_pulse_amplitudes[qubit] = platform.settings["native_gates"]["single_qubit"][
            qubit
        ]["RX"]["amplitude"]

    count = 0
    # repeat N iter times

    for iteration in range(software_averages):
        for n in range(0, niter, step):
            if count % points == 0 and count > 0:
                yield data
                yield flipping_fit(
                    data,
                    x="flips[dimensionless]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    pi_pulse_amplitude=pi_pulse_amplitudes[qubit],
                    labels=["amplitude_delta", "corrected_amplitude"],
                )

            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
                sequence.add(RX90_pulse)
                # execute sequence RX(pi/2) - [RX(pi) - RX(pi)] from 0...n times - RO
                start1 = RX90_pulse.duration
                for j in range(n):
                    RX_pulse1 = platform.create_RX_pulse(qubit, start=start1)
                    start2 = start1 + RX_pulse1.duration
                    RX_pulse2 = platform.create_RX_pulse(qubit, start=start2)
                    sequence.add(RX_pulse1)
                    sequence.add(RX_pulse2)
                    start1 = start2 + RX_pulse2.duration

                # add ro pulse at the end of the sequence
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit, start=start1
                )
                sequence.add(ro_pulses[qubit])

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "flips[dimensionless]": n,
                    "qubit": qubit,
                    "iteration": iteration,
                }
                data.add(results)
            count += 1
    yield data

    yield flipping_fit(
        data,
        x="flips[dimensionless]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        pi_pulse_amplitude=pi_pulse_amplitudes[qubit],
        labels=["amplitude_delta", "corrected_amplitude"],
    )
