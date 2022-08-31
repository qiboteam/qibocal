# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def ro_pulse_phase(
    platform,
    qubit,
    pulse_phase_start,
    pulse_phase_end,
    pulse_phase_step,
    software_averages,
    points=10,
):

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    ro_pulse_phase_range = np.arange(
        pulse_phase_start, pulse_phase_end, pulse_phase_step
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"RO_pulse_phase": "rad"})

    count = 0
    for _ in range(software_averages):
        for phase in ro_pulse_phase_range:
            if count % points == 0:
                yield data
            ro_pulse.relative_phase = phase
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "RO_pulse_phase[rad]": phase,
            }
            data.add(results)
            count += 1
    yield data
