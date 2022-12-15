import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import flipping_fit


@plot("MSR vs Flips", plots.flips_msr_phase)
def flipping(
    platform: AbstractPlatform,
    qubits: list,
    niter,
    step,
    software_averages=1,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name="data", quantities={"flips": "dimensionless"}, options=["qubit"]
    )
    pi_pulse_amplitudes = {}
    for qubit in qubits:
        pi_pulse_amplitudes["qubit"] = platform.settings["native_gates"][
            "single_qubit"
        ][qubit]["RX"]["amplitude"]

    count = 0
    # repeat N iter times
    for _ in range(software_averages):
        for n in range(0, niter, step):
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield flipping_fit(
                        data.get_column("qubit", qubit),
                        x="flips[dimensionless]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        niter=niter,
                        pi_pulse_amplitude=pi_pulse_amplitudes["qubit"],
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
                    ro_pulses["qubit"] = platform.create_qubit_readout_pulse(
                        qubit, start=start1
                    )
                    sequence.add(ro_pulses["qubit"])

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses["qubit"].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "flips[dimensionless]": n,
                    "qubit": qubit,
                }
                data.add(results)
            count += 1
    yield data
