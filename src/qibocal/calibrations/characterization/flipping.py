import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import flipping_fit


@plot("MSR vs Flips", plots.flips_msr)
def flipping(
    platform: AbstractPlatform,
    qubits: dict,
    nflips_max,
    nflips_step,
    software_averages=1,
    points=10,
):
    r"""
    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        nflips_max (int): Maximum number of flips introduced in each sequence
        nflips_step (int): Scan range step for the number of flippings
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **flips[dimensionless]**: Number of flips applied in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **amplitude_correction_factor**: Pi pulse correction factor
            - **corrected_amplitude**: Corrected pi pulse amplitude
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **popt3**: p3
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a DataUnits object to store MSR, phase, i, q and the number of flips
    data = DataUnits(
        name="data",
        quantities={"flips": "dimensionless"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for flips in range(0, nflips_max, nflips_step):
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
                # calculate and save fit
                yield flipping_fit(
                    data,
                    x="flips[dimensionless]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    pi_pulse_amplitudes={
                        q: qubits[qubit].pi_pulse_amplitude for q in qubits
                    },
                    labels=["amplitude_correction_factor", "corrected_amplitude"],
                )

            # create a sequence of pulses for the experiment
            sequence = PulseSequence()
            ro_pulses = {}
            for qubit in qubits:
                RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
                sequence.add(RX90_pulse)
                # execute sequence RX(pi/2) - [RX(pi) - RX(pi)] from 0...flips times - RO
                start1 = RX90_pulse.duration
                for j in range(flips):
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

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].raw
                r.update(
                    {
                        "flips[dimensionless]": flips,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
                data.add(r)
            count += 1
    yield data

    yield flipping_fit(
        data,
        x="flips[dimensionless]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        pi_pulse_amplitudes={q: qubits[qubit].pi_pulse_amplitude for q in qubits},
        labels=["amplitude_correction_factor", "corrected_amplitude"],
    )
