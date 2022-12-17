import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import t1_fit


@plot("MSR vs time", plots.t1_time_msr_phase)
def t1(
    platform: AbstractPlatform,
    qubits: list,
    delay_before_readout_start,
    delay_before_readout_end,
    delay_before_readout_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    sequence = PulseSequence()

    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    ro_wait_range = np.arange(
        delay_before_readout_start, delay_before_readout_end, delay_before_readout_step
    )

    data = DataUnits(name="data", quantities={"time": "ns"}, options=["qubit"])

    count = 0
    for _ in range(software_averages):
        for wait in ro_wait_range:
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield t1_fit(
                        data.get_column("qubit", qubit),
                        x="time[ns]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["t1"],
                    )

            for qubit in qubits:
                ro_pulses[qubit].start = qd_pulses[qubit].duration + wait

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "time[ns]": wait,
                    "qubit": qubit,
                }
                data.add(results)
            count += 1
    yield data
