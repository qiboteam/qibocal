import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("exc vs gnd", plots.exc_gnd)
def calibrate_qubit_states(
    platform: AbstractPlatform,
    qubits: list,
    niter,
    points=10,
):

    # create exc sequence
    exc_sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        RX_pulse = platform.create_RX_pulse(qubit, start=0)
        ro_pulses["qubit"] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulse.duration
        )
        exc_sequence.add(RX_pulse)
        exc_sequence.add(ro_pulses["qubit"])

        # FIXME: Waiting to be able to pass qpucard to qibolab
        platform.ro_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulse.frequency
        )
        platform.qd_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["qubit_freq"]
            - RX_pulse.frequency
        )

    data_exc = DataUnits(
        name="data_exc", quantities={"iteration": "s"}, options=["qubit"]
    )

    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_exc

        result = platform.execute_pulse_sequence(exc_sequence, nshots=1)

        for qubit in qubits:
            msr, phase, i, q = result[ro_pulses["qubit"].serial]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "iteration[s]": n,
                "qubit": qubit,
            }
            data_exc.add(results)
        count += 1
    yield data_exc

    gnd_sequence = PulseSequence()

    ro_pulses_gnd = {}
    for qubit in qubits:
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
        gnd_sequence.add(ro_pulse)

    data_gnd = DataUnits(
        name="data_gnd", quantities={"iteration": "s"}, options=["qubit"]
    )
    count = 0
    for n in np.arange(niter):
        if count % points == 0:
            yield data_gnd

        result = platform.execute_pulse_sequence(gnd_sequence, nshots=1)

        for qubit in qubits:
            msr, phase, i, q = result[ro_pulses_gnd["qubit"].serial]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[deg]": phase,
                "iteration[s]": n,
                "qubit": qubit,
            }
        count += 1
    yield data_gnd
