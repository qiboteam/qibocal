# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot
from qibocal.fitting.methods import flipping_fit


@plot("MSR vs Flips", plots.flips_msr_phase)
def flipping(
    platform: AbstractPlatform,
    qubit: int,
    niter,
    step,
    points=10,
):
    platform.reload_settings()
    pi_pulse_amplitude = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "amplitude"
    ]

    data = Dataset(name=f"data_q{qubit}", quantities={"flips": "dimensionless"})

    sequence = PulseSequence()
    RX90_pulse = platform.create_RX90_pulse(qubit, start=0)

    count = 0
    # repeat N iter times
    for n in range(0, niter, step):
        if count % points == 0 and count > 0:
            yield data
            yield flipping_fit(
                data,
                x="flips[dimensionless]",
                y="MSR[uV]",
                qubit=qubit,
                nqubits=platform.settings["nqubits"],
                niter=niter,
                pi_pulse_amplitude=pi_pulse_amplitude,
                labels=["amplitude_delta", "corrected_amplitude"],
            )
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
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=start1)
        sequence.add(ro_pulse)

        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "flips[dimensionless]": n,
        }
        data.add(results)
        count += 1
        sequence = PulseSequence()

    yield data
