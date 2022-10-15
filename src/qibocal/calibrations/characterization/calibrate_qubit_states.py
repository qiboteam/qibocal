# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot


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

    data_exc = Dataset(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    iq_exc = []
    count = 0
    for n in np.arange(nshots):
        if count % points == 0:
            yield data_exc
        msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots=1)[
            ro_pulse.serial
        ]
        iq_exc += [i + 1j * q]
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

    data_gnd = Dataset(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    iq_gnd = []
    count = 0
    for n in np.arange(nshots):
        if count % points == 0:
            yield data_gnd
        msr, phase, i, q = platform.execute_pulse_sequence(gnd_sequence, nshots=1)[
            ro_pulse.serial
        ]
        iq_gnd += [i + 1j * q]
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
    parameters = Dataset(
        name=f"parameters_q{qubit}",
        quantities={
            "rotation_angle": "dimensionless",  # in degrees
            "threshold": "V",
            "fidelity": "dimensionless",
            "assignment_fidelity": "dimensionless",
        },
    )

    iq_mean_exc = np.mean(iq_exc)
    iq_mean_gnd = np.mean(iq_gnd)
    origin = iq_mean_gnd

    iq_gnd_translated = iq_gnd - origin
    iq_exc_translated = iq_exc - origin
    rotation_angle = np.angle(np.mean(iq_exc_translated))

    iq_exc_rotated = iq_exc_translated * np.exp(-1j * rotation_angle) + origin
    iq_gnd_rotated = iq_gnd_translated * np.exp(-1j * rotation_angle) + origin

    real_values_exc = iq_exc_rotated.real
    real_values_gnd = iq_gnd_rotated.real

    real_values_combined = np.concatenate((real_values_exc, real_values_gnd))
    real_values_combined.sort()

    cum_distribution_exc = [
        sum(map(lambda x: x.real >= real_value, real_values_exc))
        for real_value in real_values_combined
    ]
    cum_distribution_gnd = [
        sum(map(lambda x: x.real >= real_value, real_values_gnd))
        for real_value in real_values_combined
    ]

    cum_distribution_diff = np.abs(
        np.array(cum_distribution_exc) - np.array(cum_distribution_gnd)
    )
    argmax = np.argmax(cum_distribution_diff)
    threshold = real_values_combined[argmax]
    errors_exc = nshots - cum_distribution_exc[argmax]
    errors_gnd = cum_distribution_gnd[argmax]
    fidelity = cum_distribution_diff[argmax] / nshots
    assignment_fidelity = 1 - (errors_exc + errors_gnd) / nshots / 2
    # assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2

    results = {
        "rotation_angle[dimensionless]": (rotation_angle * 360 / (2 * np.pi))
        % 360,  # in degrees
        "threshold[V]": threshold,
        "fidelity[dimensionless]": fidelity,
        "assignment_fidelity[dimensionless]": assignment_fidelity,
    }
    parameters.add(results)
    yield (parameters)


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
    data_exc = Dataset(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots)[
        "binned_integrated"
    ][ro_pulse.serial]

    iq_exc = i + 1j * q
    results = {
        "MSR[V]": msr,
        "i[V]": i,
        "q[V]": q,
        "phase[rad]": phase,
        "iteration[dimensionless]": np.arange(nshots),
    }
    data_exc.set(results)
    yield data_exc

    gnd_sequence = PulseSequence()
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )
    msr, phase, i, q = platform.execute_pulse_sequence(gnd_sequence, nshots)[
        "binned_integrated"
    ][ro_pulse.serial]

    iq_gnd = i + 1j * q
    results = {
        "MSR[V]": msr,
        "i[V]": i,
        "q[V]": q,
        "phase[rad]": phase,
        "iteration[dimensionless]": np.arange(nshots),
    }
    data_gnd.set(results)
    yield data_gnd

    parameters = Dataset(
        name=f"parameters_q{qubit}",
        quantities={
            "rotation_angle": "dimensionless",  # in degrees
            "threshold": "V",
            "fidelity": "dimensionless",
            "assignment_fidelity": "dimensionless",
        },
    )

    iq_mean_exc = np.mean(iq_exc)
    iq_mean_gnd = np.mean(iq_gnd)
    origin = iq_mean_gnd

    iq_gnd_translated = iq_gnd - origin
    iq_exc_translated = iq_exc - origin
    rotation_angle = np.angle(np.mean(iq_exc_translated))

    iq_exc_rotated = iq_exc_translated * np.exp(-1j * rotation_angle) + origin
    iq_gnd_rotated = iq_gnd_translated * np.exp(-1j * rotation_angle) + origin

    real_values_exc = iq_exc_rotated.real
    real_values_gnd = iq_gnd_rotated.real

    real_values_combined = np.concatenate((real_values_exc, real_values_gnd))
    real_values_combined.sort()

    cum_distribution_exc = [
        sum(map(lambda x: x.real >= real_value, real_values_exc))
        for real_value in real_values_combined
    ]
    cum_distribution_gnd = [
        sum(map(lambda x: x.real >= real_value, real_values_gnd))
        for real_value in real_values_combined
    ]

    cum_distribution_diff = np.abs(
        np.array(cum_distribution_exc) - np.array(cum_distribution_gnd)
    )
    argmax = np.argmax(cum_distribution_diff)
    threshold = real_values_combined[argmax]
    errors_exc = nshots - cum_distribution_exc[argmax]
    errors_gnd = cum_distribution_gnd[argmax]
    fidelity = cum_distribution_diff[argmax] / nshots
    assignment_fidelity = 1 - (errors_exc + errors_gnd) / nshots / 2
    # assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2

    results = {
        "rotation_angle[dimensionless]": (rotation_angle * 360 / (2 * np.pi))
        % 360,  # in degrees
        "threshold[V]": threshold,
        "fidelity[dimensionless]": fidelity,
        "assignment_fidelity[dimensionless]": assignment_fidelity,
    }
    parameters.add(results)
    yield (parameters)
