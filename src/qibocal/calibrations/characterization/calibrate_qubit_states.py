# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("exc vs gnd", plots.exc_gnd)
def calibrate_qubit_states(
    platform: AbstractPlatform,
    qubit: int,
    niter,
    points=10,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubit (int): index representing the target qubit into the chip.
        niter (int): number of times the pulse sequence will be reproduced.
        points (int): every points step data are saved.  
    """

    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)

    # FIXME: Waiting to be able to pass qpucard to qibolab
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - RX_pulse.frequency
    )

    data_exc = DataUnits(name=f"data_exc_q{qubit}", quantities={"iteration": "s"})

    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_exc
        msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots=1)[
            ro_pulse.serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
            "iteration[s]": n,
        }
        data_exc.add(results)
        count += 1
    yield data_exc

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    gnd_sequence.add(ro_pulse)

    data_gnd = DataUnits(name=f"data_gnd_q{qubit}", quantities={"iteration": "s"})
    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_gnd

        msr, phase, i, q = platform.execute_pulse_sequence(gnd_sequence, nshots=1)[
            ro_pulse.serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
            "iteration[s]": n,
        }
        data_gnd.add(results)
        count += 1
    yield data_gnd
