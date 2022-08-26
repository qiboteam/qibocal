# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def calibrate_qubit_states(
    platform,
    qubit,
    niter,
    points=10,
):

    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.RX_pulse(qubit, start=0)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=RX_pulse.duration)
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

    data_exc = Dataset(name=f"data_exc_q{qubit}", quantities={"iteration": "s"})

    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_exc
        msr, i, q, phase = platform.execute_pulse_sequence(exc_sequence, nshots=1)[0][
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
    ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(name=f"data_gnd_q{qubit}", quantities={"iteration": "s"})
    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_gnd
        msr, i, q, phase = platform.execute_pulse_sequence(gnd_sequence, nshots=1)[0][
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
