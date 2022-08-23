# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def t1(
    platform,
    qubit,
    delay_before_readout_start,
    delay_before_readout_end,
    delay_before_readout_step,
    software_averages,
    points=10,
):

    sequence = PulseSequence()
    qd_pulse = platform.RX_pulse(qubit, start=0)
    ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_before_readout_start, delay_before_readout_end, delay_before_readout_step
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0:
                yield data
            ro_pulse.start = qd_pulse.duration + wait
            msr, i, q, phase = platform.execute_pulse_sequence(sequence)[0][
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "Time[ns]": wait,
            }
            data.add(results)
            count += 1
    yield data
