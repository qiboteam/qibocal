# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations import utils
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def flipping(
    platform,
    qubit,
    niter,
    step,
    points=10,
):
    platform.reload_settings()

    sequence = PulseSequence()
    RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
    res = []
    N = []

    data = Dataset(name=f"data_q{qubit}", quantities={"flips": "dimensionless"})

    count = 0
    # repeat N iter times
    for n in range(0, niter, step):
        if count % points == 0:
            yield data
        # execute sequence RX(pi/2) - [RX(pi) - Rx(pi)] from 0...i times - RO
        sequence.add(RX90_pulse)
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
            "flips[dimensionless]": np.array(n),
        }
        data.add(results)
        sequence = PulseSequence()
        res += [msr]
        N += [i]
    yield data

    # Fitting results to obtain epsilon
    if platform.resonator_type == "3D":
        popt = utils.flipping_fit_3D(N, res)
    elif platform.resonator_type == "2D":
        popt = utils.flipping_fit_2D(N, res)

    angle = (niter * 2 * np.pi / popt[2] + popt[3]) / (1 + 4 * niter)
    state1_voltage = (
        1e-6
        * platform.settings["characterization"]["single_qubit"][qubit]["state1_voltage"]
    )
    state0_voltage = (
        1e-6
        * platform.settings["characterization"]["single_qubit"][qubit]["state0_voltage"]
    )
    pi_pulse_amplitude = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "amplitude"
    ]
    amplitude_delta = angle * 2 / np.pi * pi_pulse_amplitude
    print(amplitude_delta)
