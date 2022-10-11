import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot


@plot("exc vs gnd", plots.exc_gnd)
def calibrate_qubit_states_binning(
    platform: AbstractPlatform,
    qubit: int,
    niter,
    points=10,
):
    platform.reload_settings()
    platform.qrm[qubit].ports[
        "i1"
    ].hardware_demod_en = True  # binning only works with hardware demodulation enabled
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)
    data_exc = Dataset(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    shots_results = platform.execute_pulse_sequence(exc_sequence, nshots=niter)[
        "shots"
    ][ro_pulse.serial]
    for n in np.arange(niter):
        msr, phase, i, q = shots_results[n]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        data_exc.add(results)
    yield data_exc

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )

    shots_results = platform.execute_pulse_sequence(gnd_sequence, nshots=niter)[
        "shots"
    ][ro_pulse.serial]
    for n in np.arange(niter):
        msr, phase, i, q = shots_results[n]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        data_gnd.add(results)
    yield data_gnd
