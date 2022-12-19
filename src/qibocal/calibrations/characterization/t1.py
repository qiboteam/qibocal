import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import t1_fit


@plot("MSR vs Time", plots.t1_time_msr_phase)
def t1(
    platform: AbstractPlatform,
    qubit: int,
    delay_before_readout_start,
    delay_before_readout_end,
    delay_before_readout_step,
    software_averages,
    points=10,
):

    r"""
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "wait[ns]": Delay before ReadOut used in the current execution

        A DataUnits object with the fitted data obtained with the following keys:
            - *labels[0]*: T1
            - *popt0*: p0
            - *popt1*: p1
            - *popt2*: p2
    """
    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.duration)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    ro_wait_range = np.arange(
        delay_before_readout_start, delay_before_readout_end, delay_before_readout_step
    )

    data = DataUnits(name=f"data_q{qubit}", quantities={"Time": "ns"})

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0 and count > 0:
                yield data
                yield t1_fit(
                    data,
                    x="Time[ns]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["t1"],
                )
            ro_pulse.start = qd_pulse.duration + wait
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "Time[ns]": wait,
            }
            data.add(results)
            count += 1
    yield data
