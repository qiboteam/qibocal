import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import calibrate_qubit_state_fit


@plot("exc vs gnd", plots.exc_gnd)
def calibrate_qubit_states(
    platform: AbstractPlatform,
    qubit: int,
    nshots,
    points=10,
):
    platform.reload_settings()
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)

    data_exc = DataUnits(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    count = 0
    for n in np.arange(nshots):
        if count % points == 0:
            yield data_exc
        msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots=1)[
            ro_pulse.serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        data_exc.add(results)
        count += 1
    yield data_exc

    gnd_sequence = PulseSequence()
    gnd_sequence.add(ro_pulse)

    data_gnd = DataUnits(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    count = 0
    for n in np.arange(nshots):
        if count % points == 0:
            yield data_gnd
        msr, phase, i, q = platform.execute_pulse_sequence(gnd_sequence, nshots=1)[
            ro_pulse.serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        data_gnd.add(results)
        count += 1
    yield data_gnd
    yield calibrate_qubit_state_fit(
        data_gnd, data_exc, x="i[V]", y="q[V]", nshots=nshots, qubit=qubit
    )


@plot("exc vs gnd", plots.exc_gnd)
def calibrate_qubit_states_binning(
    platform: AbstractPlatform,
    qubit: int,
    nshots,
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
    data_exc = DataUnits(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots)[
        "binned_integrated"
    ][ro_pulse.serial]

    results = {
        "MSR[V]": msr,
        "i[V]": i,
        "q[V]": q,
        "phase[rad]": phase,
        "iteration[dimensionless]": np.arange(nshots),
    }
    data_exc.load_data_from_dict(results)
    yield data_exc

    gnd_sequence = PulseSequence()
    gnd_sequence.add(ro_pulse)

    data_gnd = DataUnits(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    msr, phase, i, q = platform.execute_pulse_sequence(gnd_sequence, nshots)[
        "binned_integrated"
    ][ro_pulse.serial]

    results = {
        "MSR[V]": msr,
        "i[V]": i,
        "q[V]": q,
        "phase[rad]": phase,
        "iteration[dimensionless]": np.arange(nshots),
    }
    data_gnd.load_data_from_dict(results)
    yield data_gnd
    yield calibrate_qubit_state_fit(
        data_gnd, data_exc, x="i[V]", y="q[V]", nshots=nshots, qubit=qubit
    )
